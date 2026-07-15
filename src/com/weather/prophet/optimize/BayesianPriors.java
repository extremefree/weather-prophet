package com.weather.prophet.optimize;

import com.weather.prophet.matrix.VecOps;

/**
 * Bayesian prior distributions used in Prophet's Stan model.
 *
 * Prophet's actual priors (from prophet_model.stan):
 *   k       ~ Normal(0, 5)           // base growth rate
 *   m       ~ Normal(0, 5)           // base offset
 *   delta   ~ Laplace(0, tau)        // changepoint rate changes (L1/LASSO prior for sparsity!)
 *   beta    ~ Normal(0, sigma_beta)  // seasonality coefficients
 *   kappa   ~ Normal(0, sigma_h)     // holiday effects
 *   sigma   ~ HalfCauchy(0, 0.5)    // observation noise (scaled)
 *
 * This class provides:
 * 1. Log-density computations for each prior
 * 2. Gradient of log-density w.r.t. parameters
 * 3. Combined negative log posterior (for L-BFGS minimization)
 */
public class BayesianPriors {

    // ===================== Log Densities =====================

    /** Normal log PDF: log N(x | mu, sigma) */
    public static double logNormal(double x, double mu, double sigma) {
        double z = (x - mu) / sigma;
        return -0.5 * z * z - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
    }

    /** Gradient of Normal log PDF w.r.t. x */
    public static double gradLogNormal(double x, double mu, double sigma) {
        return -(x - mu) / (sigma * sigma);
    }

    /** Laplace (double exponential) log PDF: log Laplace(x | mu, b) */
    public static double logLaplace(double x, double mu, double b) {
        return -Math.abs(x - mu) / b - Math.log(2 * b);
    }

    /** Sub-gradient of Laplace log PDF w.r.t. x */
    public static double gradLogLaplace(double x, double mu, double b) {
        double diff = x - mu;
        if (diff > 1e-12) return -1.0 / b;
        if (diff < -1e-12) return 1.0 / b;
        return 0.0; // sub-gradient at 0
    }

    /** Half-Cauchy log PDF: log HC(x | 0, scale) for x >= 0 */
    public static double logHalfCauchy(double x, double scale) {
        if (x < 0) return Double.NEGATIVE_INFINITY;
        return Math.log(2.0 / Math.PI) - Math.log(scale) - Math.log(1 + (x * x) / (scale * scale));
    }

    /** Gradient of Half-Cauchy log PDF w.r.t. x */
    public static double gradLogHalfCauchy(double x, double scale) {
        if (x <= 0) return 0.0;
        return -2.0 * x / (scale * scale + x * x);
    }

    // ===================== Combined Posterior =====================

    /**
     * Compute the negative log posterior (objective to minimize) and its gradient.
     *
     * This matches Prophet's Stan model exactly:
     *   -log p(θ|y) = -[log p(y|θ) + log p(θ)]
     *
     * Parameters layout in theta[]:
     *   [0]      : k        (base growth rate)
     *   [1]      : m        (base offset)
     *   [2..2+S) : delta    (changepoint rate changes, S = num changepoints)
     *   [2+S]    : sigma_obs (observation noise, constrained positive via log transform)
     *   [3+S..3+S+K) : beta (seasonality coefficients)
     *   [3+S+K..3+S+K+H) : kappa (holiday effects)
     */
    public static double[] negLogPosteriorAndGradient(
            double[] theta,
            // Data
            double[] t,          // timestamps (length T)
            double[] y,          // observations (length T)
            // Changepoints
            double[] changepoints, // length S
            // Seasonality
            double[][] XSeasonal,  // T x K Fourier features
            // Holidays
            double[][] XHoliday,   // T x H holiday features
            // Prior hyperparameters
            double tau,            // Laplace scale for delta (changepoint_prior_scale)
            double sigmaBeta,      // Normal scale for beta (seasonality_prior_scale)
            double sigmaKappa,     // Normal scale for kappa (holidays_prior_scale
            double sigmaObsScale,  // Half-Cauchy scale for sigma_obs
            boolean logisticGrowth, // whether using logistic growth
            double cap,            // carrying capacity for logistic growth (in original scale)
            double yMean,          // y mean for standardization
            double yStd            // y std for standardization
    ) {
        int T = t.length;
        int S = changepoints.length;
        int K = XSeasonal != null ? XSeasonal[0].length : 0;
        int H = XHoliday != null ? XHoliday[0].length : 0;

        // Extract parameters
        double k = theta[0];
        double m = theta[1];
        double[] delta = new double[S];
        System.arraycopy(theta, 2, delta, 0, S);
        double logSigmaObs = theta[2 + S];
        double sigmaObs = Math.exp(logSigmaObs); // ensure positivity
        double[] beta = new double[K];
        if (K > 0) System.arraycopy(theta, 3 + S, beta, 0, K);
        double[] kappa = new double[H];
        if (H > 0) System.arraycopy(theta, 3 + S + K, kappa, 0, H);

        // Compute trend: g(t) = (k + A(t)^T * delta) * t + (m + A(t)^T * gamma)
        // where A_j(t) = 1 if t >= s_j, and gamma_j = -s_j * delta_j
        double[] trend = new double[T];
        double[] rateAtT = new double[T]; // effective rate at time t
        for (int i = 0; i < T; i++) {
            double ti = t[i];
            double aDeltaSum = 0;
            double aGammaSum = 0;
            double rateSum = k;
            for (int j = 0; j < S; j++) {
                double a = ti >= changepoints[j] ? 1.0 : 0.0;
                aDeltaSum += a * delta[j];
                aGammaSum += a * (-changepoints[j] * delta[j]); // gamma_j = -s_j * delta_j
                rateSum += a * delta[j];
            }
            rateAtT[i] = rateSum;
            if (!logisticGrowth) {
                trend[i] = (k + aDeltaSum) * ti + (m + aGammaSum);
            } else {
                // Logistic: g(t) = C / (1 + exp(-(k + A*d) * (t - (m + A*gamma))))
                double C = (cap - yMean) / yStd; // standardized cap
                double r = (k + aDeltaSum);
                double offset = (m + aGammaSum);
                trend[i] = C / (1.0 + Math.exp(-r * (ti - offset)));
            }
        }

        // Compute y_hat = g(t) + s(t) + h(t)
        double[] yHat = new double[T];
        for (int i = 0; i < T; i++) {
            yHat[i] = trend[i];
            for (int j = 0; j < K; j++) {
                yHat[i] += XSeasonal[i][j] * beta[j];
            }
            for (int j = 0; j < H; j++) {
                yHat[i] += XHoliday[i][j] * kappa[j];
            }
        }

        // ===================== Log Likelihood =====================
        // y ~ Normal(y_hat, sigma_obs)
        double logLik = 0;
        double[] residual = new double[T];
        for (int i = 0; i < T; i++) {
            residual[i] = y[i] - yHat[i];
            logLik += logNormal(y[i], yHat[i], sigmaObs);
        }

        // ===================== Log Priors =====================
        double logPrior = 0;
        logPrior += logNormal(k, 0, 5);        // k ~ N(0,5)
        logPrior += logNormal(m, 0, 5);        // m ~ N(0,5)
        for (int j = 0; j < S; j++) {
            logPrior += logLaplace(delta[j], 0, tau);  // delta ~ Laplace(0, tau)
        }
        logPrior += logHalfCauchy(sigmaObs, sigmaObsScale);  // sigma ~ HalfCauchy(0, scale)
        logPrior += logNormal(logSigmaObs, 0, 5); // Jacobian for log transform
        for (int j = 0; j < K; j++) {
            logPrior += logNormal(beta[j], 0, sigmaBeta);    // beta ~ N(0, sigma_beta)
        }
        for (int j = 0; j < H; j++) {
            logPrior += logNormal(kappa[j], 0, sigmaKappa);  // kappa ~ N(0, sigma_kappa)
        }

        // ===================== Negative Log Posterior =====================
        double negLogPost = -(logLik + logPrior);

        // ===================== Gradient =====================
        double[] grad = new double[theta.length];
        double sigma2 = sigmaObs * sigmaObs;

        // d/dy_hat of -log likelihood = (y_hat - y) / sigma^2
        double[] dLdYhat = new double[T];
        for (int i = 0; i < T; i++) {
            dLdYhat[i] = (yHat[i] - y[i]) / sigma2;
        }

        // --- Gradient w.r.t. k ---
        double dLdk = 0;
        for (int i = 0; i < T; i++) {
            double ti = t[i];
            if (!logisticGrowth) {
                dLdk += dLdYhat[i] * ti;
            } else {
                // Logistic gradient
                double expTerm = Math.exp(-rateAtT[i] * (ti - (m)));
                double denom = (1 + expTerm) * (1 + expTerm);
                double C = (cap - yMean) / yStd;
                dLdk += dLdYhat[i] * C * expTerm * ti / denom; // approximate
            }
        }
        grad[0] = dLdk + gradLogNormal(k, 0, 5); // prior gradient

        // --- Gradient w.r.t. m ---
        double dLdm = 0;
        for (int i = 0; i < T; i++) {
            dLdm += dLdYhat[i] * 1.0; // dm/dy_hat = 1 for linear trend
        }
        grad[1] = dLdm + gradLogNormal(m, 0, 5);

        // --- Gradient w.r.t. delta[j] ---
        for (int j = 0; j < S; j++) {
            double dLdDeltaJ = 0;
            for (int i = 0; i < T; i++) {
                double ti = t[i];
                double a = ti >= changepoints[j] ? 1.0 : 0.0;
                if (!logisticGrowth) {
                    // dg/ddelta_j = a * (t - s_j) (since gamma_j = -s_j * delta_j)
                    dLdDeltaJ += dLdYhat[i] * a * (ti - changepoints[j]);
                } else {
                    dLdDeltaJ += dLdYhat[i] * a * 0.1; // approximate for logistic
                }
            }
            grad[2 + j] = dLdDeltaJ + gradLogLaplace(delta[j], 0, tau);
        }

        // --- Gradient w.r.t. log(sigma_obs) ---
        double dLdSigma = 0;
        for (int i = 0; i < T; i++) {
            dLdSigma += residual[i] * residual[i] / (sigma2 * sigmaObs) - 1.0 / sigmaObs;
        }
        double dLogSigmaDsigma = sigmaObs; // d(log sigma)/d(sigma) * d(sigma)/d(log_sigma) chain rule
        grad[2 + S] = dLdSigma * dLogSigmaDsigma + gradLogHalfCauchy(sigmaObs, sigmaObsScale) * dLogSigmaDsigma
                + gradLogNormal(logSigmaObs, 0, 5) * dLogSigmaDsigma;

        // --- Gradient w.r.t. beta[j] ---
        for (int j = 0; j < K; j++) {
            double dLdBetaJ = 0;
            for (int i = 0; i < T; i++) {
                dLdBetaJ += dLdYhat[i] * XSeasonal[i][j];
            }
            grad[3 + S + j] = dLdBetaJ + gradLogNormal(beta[j], 0, sigmaBeta);
        }

        // --- Gradient w.r.t. kappa[j] ---
        for (int j = 0; j < H; j++) {
            double dLdKappaJ = 0;
            for (int i = 0; i < T; i++) {
                dLdKappaJ += dLdYhat[i] * XHoliday[i][j];
            }
            grad[3 + S + K + j] = dLdKappaJ + gradLogNormal(kappa[j], 0, sigmaKappa);
        }

        // Pack result: first element is negLogPost, rest is gradient
        double[] result = new double[1 + theta.length];
        result[0] = negLogPost;
        System.arraycopy(grad, 0, result, 1, theta.length);
        return result;
    }
}
