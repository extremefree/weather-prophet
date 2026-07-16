package com.weather.prophet.optimize;

import com.weather.prophet.matrix.VecOps;

/**
 * Bayesian prior distributions and posterior — faithful to Prophet's Stan model.
 *
 * Reference: facebook/prophet python/stan/prophet.stan
 *
 * Stan model:
 *   k       ~ Normal(0, 5)
 *   m       ~ Normal(0, 5)
 *   delta   ~ double_exponential(0, tau)   // Laplace
 *   sigma_obs ~ Normal(0, 0.5)             // truncated to positive (lower=0)
 *   beta    ~ Normal(0, sigmas)            // per-component scale vector
 *
 * Likelihood (Stan normal_id_glm):
 *   y ~ Normal(trend .* (1 + X_sm * beta) + X_sa * beta, sigma_obs)
 *
 * Logistic gamma (Stan logistic_gamma function):
 *   k_s = [k, k+δ₁, k+δ₁+δ₂, ...]
 *   m_pr = m
 *   for i in 1..S:
 *     gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i+1])
 *     m_pr += gamma[i]
 *
 * Linear gamma:
 *   gamma[i] = -t_change[i] * delta[i]
 *
 * Flat trend:
 *   trend[i] = m  (constant for all i)
 */
public class BayesianPriors {

    // ===================== Log Densities =====================

    public static double logNormal(double x, double mu, double sigma) {
        double z = (x - mu) / sigma;
        return -0.5 * z * z - Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
    }

    public static double gradLogNormal(double x, double mu, double sigma) {
        return -(x - mu) / (sigma * sigma);
    }

    public static double logLaplace(double x, double mu, double b) {
        return -Math.abs(x - mu) / b - Math.log(2 * b);
    }

    public static double gradLogLaplace(double x, double mu, double b) {
        double diff = x - mu;
        if (diff > 1e-12) return -1.0 / b;
        if (diff < -1e-12) return 1.0 / b;
        return 0.0;
    }

    // ===================== Trend Functions =====================

    /**
     * Compute logistic gamma (Stan logistic_gamma function).
     * Reference: prophet.stan line 31-46
     */
    public static double[] logisticGamma(double k, double m, double[] delta, double[] changepoints) {
        int S = changepoints.length;
        double[] gamma = new double[S];
        double[] k_s = new double[S + 1];
        k_s[0] = k;
        for (int i = 0; i < S; i++) {
            k_s[i + 1] = k_s[i] + delta[i];
        }
        double m_pr = m;
        for (int i = 0; i < S; i++) {
            gamma[i] = (changepoints[i] - m_pr) * (1.0 - k_s[i] / k_s[i + 1]);
            m_pr += gamma[i];
        }
        return gamma;
    }

    /**
     * Compute linear gamma: gamma_j = -s_j * delta_j
     * Reference: prophet.stan line 74
     */
    public static double[] linearGamma(double[] delta, double[] changepoints) {
        int S = changepoints.length;
        double[] gamma = new double[S];
        for (int i = 0; i < S; i++) {
            gamma[i] = -changepoints[i] * delta[i];
        }
        return gamma;
    }

    /**
     * Compute trend for all growth types.
     * Reference: prophet.stan line 118-126
     *
     * @param growthType 0=linear, 1=logistic, 2=flat
     */
    public static double[] computeTrend(
            double k, double m, double[] delta,
            double[] t, double[] changepoints,
            int growthType, double[] capScaled) {

        int T = t.length;
        int S = changepoints.length;
        double[] trend = new double[T];

        if (growthType == 2) {
            // Flat trend: trend[i] = m
            for (int i = 0; i < T; i++) trend[i] = m;
            return trend;
        }

        // Compute gamma
        double[] gamma;
        if (growthType == 1) {
            gamma = logisticGamma(k, m, delta, changepoints);
        } else {
            gamma = linearGamma(delta, changepoints);
        }

        // Compute A matrix (changepoint indicator) and trend
        for (int i = 0; i < T; i++) {
            double ti = t[i];
            double aDelta = 0, aGamma = 0;
            for (int j = 0; j < S; j++) {
                if (ti >= changepoints[j]) {
                    aDelta += delta[j];
                    aGamma += gamma[j];
                }
            }
            double r = k + aDelta;
            double offset = m + aGamma;

            if (growthType == 0) {
                // Linear: trend = (k + A*delta) * t + (m + A*gamma)
                trend[i] = r * ti + offset;
            } else {
                // Logistic: trend = cap / (1 + exp(-r * (t - offset)))
                double cap_i = capScaled != null ? capScaled[i] : 1.0;
                double u = r * (ti - offset);
                // Clamp u to prevent overflow
                if (u > 500) u = 500;
                if (u < -500) u = -500;
                trend[i] = cap_i / (1.0 + Math.exp(-u));
            }
        }
        return trend;
    }

    // ===================== Combined Posterior =====================

    /**
     * Compute the negative log posterior (objective to minimize) and its gradient.
     *
     * Parameters layout in theta[]:
     *   [0]          : k        (base growth rate; not used for flat)
     *   [1]          : m        (base offset)
     *   [2..2+S)     : delta    (changepoint rate changes)
     *   [2+S]        : logSigma (log of observation noise)
     *   [3+S..3+S+K) : beta     (seasonality + holiday coefficients)
     *
     * @param sigmas     per-component prior scales (length K), matching Prophet's vector sigmas
     * @param s_a        additive indicator vector (length K), 1.0 if additive, 0.0 otherwise
     * @param s_m        multiplicative indicator vector (length K), 1.0 if multiplicative
     * @param growthType 0=linear, 1=logistic, 2=flat
     * @param capScaled  standardized cap (length T), for logistic
     * @param yScale     absmax scaling factor (for reference, not used in computation since y is already scaled)
     */
    public static double[] negLogPosteriorAndGradient(
            double[] theta,
            double[] t, double[] y,
            double[] changepoints,
            double[][] X,          // T x K all regressor features (seasonal + holiday)
            double[] sigmas,        // length K, per-component prior scales
            double[] s_a,           // length K, additive indicators
            double[] s_m,           // length K, multiplicative indicators
            double tau,             // Laplace scale for delta
            double sigmaObsPriorScale,  // Normal scale for sigma_obs (Stan: normal(0, 0.5))
            int growthType,         // 0=linear, 1=logistic, 2=flat
            double[] capScaled      // length T, standardized cap for logistic; null otherwise
    ) {
        int T = t.length;
        int S = changepoints.length;
        int K = X != null ? X[0].length : 0;

        // Extract parameters
        double k = theta[0];
        double m = theta[1];
        double[] delta = new double[S];
        System.arraycopy(theta, 2, delta, 0, S);
        double logSigma = theta[2 + S];
        double sigmaObs = Math.exp(logSigma);
        double[] beta = new double[K];
        if (K > 0) System.arraycopy(theta, 3 + S, beta, 0, K);

        // Compute trend
        double[] trend = computeTrend(k, m, delta, t, changepoints, growthType, capScaled);

        // Compute y_hat = trend * (1 + X_sm * beta) + X_sa * beta
        // Reference: Stan line 137-142
        double[] yHat = new double[T];
        double[] trendMult = new double[T];  // dy_hat/d(trend) = 1 + X_sm * beta
        for (int i = 0; i < T; i++) {
            double multPart = 0.0;  // X_sm[i] · beta
            double addPart = 0.0;   // X_sa[i] · beta
            for (int j = 0; j < K; j++) {
                multPart += X[i][j] * s_m[j] * beta[j];
                addPart += X[i][j] * s_a[j] * beta[j];
            }
            trendMult[i] = 1.0 + multPart;
            yHat[i] = trend[i] * trendMult[i] + addPart;
        }

        // Log likelihood: y ~ Normal(y_hat, sigma_obs)
        double logLik = 0;
        double[] residual = new double[T];
        for (int i = 0; i < T; i++) {
            residual[i] = y[i] - yHat[i];
            logLik += logNormal(y[i], yHat[i], sigmaObs);
        }

        // Log priors (matching Stan exactly)
        double logPrior = 0;
        if (growthType != 2) {
            logPrior += logNormal(k, 0, 5);   // k ~ N(0,5) (not used for flat)
        }
        logPrior += logNormal(m, 0, 5);       // m ~ N(0,5)
        for (int j = 0; j < S; j++) {
            logPrior += logLaplace(delta[j], 0, tau);  // delta ~ Laplace(0, tau)
        }
        // sigma_obs ~ Normal(0, sigmaObsPriorScale) truncated to positive
        logPrior += logNormal(sigmaObs, 0, sigmaObsPriorScale);
        // Jacobian for log transform: log|d sigma/d logSigma| = log(sigma) = logSigma
        logPrior += logSigma;

        for (int j = 0; j < K; j++) {
            logPrior += logNormal(beta[j], 0, sigmas[j]);  // beta ~ N(0, sigmas[j])
        }

        double negLogPost = -(logLik + logPrior);

        // ===================== Gradient =====================
        double[] grad = new double[theta.length];
        double sigma2 = sigmaObs * sigmaObs;

        // d(negLogLik)/d(yHat[i]) = (yHat[i] - y[i]) / sigma^2
        double[] dLdYhat = new double[T];
        for (int i = 0; i < T; i++) {
            dLdYhat[i] = (yHat[i] - y[i]) / sigma2;
        }

        // Chain rule: d(negLogLik)/d(trend[i]) = dLdYhat[i] * trendMult[i]
        double[] dLdTrend = new double[T];
        for (int i = 0; i < T; i++) {
            dLdTrend[i] = dLdYhat[i] * trendMult[i];
        }

        if (growthType == 0) {
            // === Linear trend gradients (analytic) ===
            // dtrend/d(k) = t[i]
            double dLdk = 0;
            for (int i = 0; i < T; i++) dLdk += dLdTrend[i] * t[i];
            grad[0] = dLdk - gradLogNormal(k, 0, 5);

            // dtrend/d(m) = 1
            double dLdm = 0;
            for (int i = 0; i < T; i++) dLdm += dLdTrend[i];
            grad[1] = dLdm - gradLogNormal(m, 0, 5);

            // dtrend/d(delta_j) = A[i][j] * (t[i] - s_j)  (using gamma = -s*delta)
            for (int j = 0; j < S; j++) {
                double dLdDeltaJ = 0;
                for (int i = 0; i < T; i++) {
                    if (t[i] >= changepoints[j]) {
                        dLdDeltaJ += dLdTrend[i] * (t[i] - changepoints[j]);
                    }
                }
                grad[2 + j] = dLdDeltaJ - gradLogLaplace(delta[j], 0, tau);
            }
        } else if (growthType == 2) {
            // === Flat trend gradients ===
            // trend[i] = m, so dtrend/d(m) = 1, dtrend/d(k) = 0, dtrend/d(delta) = 0
            double dLdm = 0;
            for (int i = 0; i < T; i++) dLdm += dLdTrend[i];
            grad[1] = dLdm - gradLogNormal(m, 0, 5);
            // k is not used, set gradient to 0 (or use prior to push toward 0)
            grad[0] = -gradLogNormal(k, 0, 5);
            // delta gradients: only prior, no likelihood contribution
            for (int j = 0; j < S; j++) {
                grad[2 + j] = -gradLogLaplace(delta[j], 0, tau);
            }
        } else {
            // === Logistic trend gradients (numerical for trend params) ===
            // Use finite differences for k, m, delta because logistic_gamma makes
            // the analytic gradient very complex (gamma depends on all previous deltas)
            double h = 1e-5;
            // Gradient w.r.t. k
            double[] thetaP = theta.clone(), thetaM = theta.clone();
            thetaP[0] += h; thetaM[0] -= h;
            double valP = computeNegLogPost(thetaP, t, y, changepoints, X, sigmas, s_a, s_m,
                    tau, sigmaObsPriorScale, growthType, capScaled, K, S);
            double valM = computeNegLogPost(thetaM, t, y, changepoints, X, sigmas, s_a, s_m,
                    tau, sigmaObsPriorScale, growthType, capScaled, K, S);
            grad[0] = (valP - valM) / (2 * h);

            // Gradient w.r.t. m
            thetaP = theta.clone(); thetaM = theta.clone();
            thetaP[1] += h; thetaM[1] -= h;
            valP = computeNegLogPost(thetaP, t, y, changepoints, X, sigmas, s_a, s_m,
                    tau, sigmaObsPriorScale, growthType, capScaled, K, S);
            valM = computeNegLogPost(thetaM, t, y, changepoints, X, sigmas, s_a, s_m,
                    tau, sigmaObsPriorScale, growthType, capScaled, K, S);
            grad[1] = (valP - valM) / (2 * h);

            // Gradient w.r.t. delta[j]
            for (int j = 0; j < S; j++) {
                thetaP = theta.clone(); thetaM = theta.clone();
                thetaP[2 + j] += h; thetaM[2 + j] -= h;
                valP = computeNegLogPost(thetaP, t, y, changepoints, X, sigmas, s_a, s_m,
                        tau, sigmaObsPriorScale, growthType, capScaled, K, S);
                valM = computeNegLogPost(thetaM, t, y, changepoints, X, sigmas, s_a, s_m,
                        tau, sigmaObsPriorScale, growthType, capScaled, K, S);
                grad[2 + j] = (valP - valM) / (2 * h);
            }
        }

        // === Gradient w.r.t. log(sigma_obs) ===
        // negLogLik = sum(0.5*(y-yHat)^2/sigma^2 + log(sigma) + 0.5*log(2π))
        // d(negLogLik)/d(logSigma) = -sum(residual^2/sigma^2) + T
        double sumResid2 = 0;
        for (int i = 0; i < T; i++) sumResid2 += residual[i] * residual[i];
        double dNegLogLikDLogSigma = -sumResid2 / sigma2 + T;
        // Prior: sigma_obs ~ Normal(0, sigmaPrior) + Jacobian
        // d(logPrior)/d(logSigma) = gradLogNormal(sigma, 0, sigmaPrior) * sigma + 1
        // d(-logPrior)/d(logSigma) = -(gradLogNormal(sigma, 0, sigmaPrior) * sigma + 1)
        double dNegLogPriorDLogSigma = -(gradLogNormal(sigmaObs, 0, sigmaObsPriorScale) * sigmaObs + 1.0);
        grad[2 + S] = dNegLogLikDLogSigma + dNegLogPriorDLogSigma;

        // === Gradient w.r.t. beta[j] ===
        // dy_hat/d(beta_j) = X[i][j] * (s_a[j] + s_m[j] * trend[i])
        for (int j = 0; j < K; j++) {
            double dLdBetaJ = 0;
            for (int i = 0; i < T; i++) {
                dLdBetaJ += dLdYhat[i] * X[i][j] * (s_a[j] + s_m[j] * trend[i]);
            }
            grad[3 + S + j] = dLdBetaJ - gradLogNormal(beta[j], 0, sigmas[j]);
        }

        // Pack result
        double[] result = new double[1 + theta.length];
        result[0] = negLogPost;
        System.arraycopy(grad, 0, result, 1, theta.length);
        return result;
    }

    /**
     * Compute just the negative log posterior (no gradient).
     * Used for numerical gradient computation in logistic mode.
     */
    private static double computeNegLogPost(
            double[] theta,
            double[] t, double[] y,
            double[] changepoints,
            double[][] X,
            double[] sigmas,
            double[] s_a, double[] s_m,
            double tau, double sigmaObsPriorScale,
            int growthType, double[] capScaled,
            int K, int S) {

        double k = theta[0];
        double m = theta[1];
        double[] delta = new double[S];
        System.arraycopy(theta, 2, delta, 0, S);
        double sigmaObs = Math.exp(theta[2 + S]);
        double[] beta = new double[K];
        if (K > 0) System.arraycopy(theta, 3 + S, beta, 0, K);

        double[] trend = computeTrend(k, m, delta, t, changepoints, growthType, capScaled);

        double logLik = 0;
        for (int i = 0; i < t.length; i++) {
            double multPart = 0, addPart = 0;
            for (int j = 0; j < K; j++) {
                multPart += X[i][j] * s_m[j] * beta[j];
                addPart += X[i][j] * s_a[j] * beta[j];
            }
            double yHat = trend[i] * (1.0 + multPart) + addPart;
            logLik += logNormal(y[i], yHat, sigmaObs);
        }

        double logPrior = 0;
        if (growthType != 2) logPrior += logNormal(k, 0, 5);
        logPrior += logNormal(m, 0, 5);
        for (int j = 0; j < S; j++) logPrior += logLaplace(delta[j], 0, tau);
        logPrior += logNormal(sigmaObs, 0, sigmaObsPriorScale);
        logPrior += Math.log(sigmaObs);  // Jacobian
        for (int j = 0; j < K; j++) logPrior += logNormal(beta[j], 0, sigmas[j]);

        return -(logLik + logPrior);
    }
}
