package com.weather.prophet.core;

import com.weather.prophet.data.DataPoint;
import com.weather.prophet.matrix.Matrix;
import com.weather.prophet.matrix.VecOps;
import com.weather.prophet.optimize.BayesianPriors;
import com.weather.prophet.optimize.AdamOptimizer;
import com.weather.prophet.optimize.LBFGSOptimizer;
import com.weather.prophet.mcmc.MCMCSampler;
import com.weather.prophet.compute.CPUBackend;

import java.util.*;

/**
 * Faithful Java port of Facebook Prophet.
 *
 * Key references:
 *   - python/prophet/forecaster.py  (Prophet class)
 *   - python/stan/prophet.stan       (Stan model)
 *
 * y(t) = g(t) + s(t) + h(t) + ε         (additive)
 * y(t) = g(t) * (1 + s(t) + h(t)) + ε   (multiplicative)
 *
 * where:
 *   g(t) = trend (linear / logistic / flat)
 *   s(t) = seasonality (Fourier series)
 *   h(t) = holiday effects
 *   ε    ~ Normal(0, sigma_obs)
 */
public class ProphetModel {

    private final ProphetConfig config;

    // Training data (standardized)
    private double[] trainT;       // scaled time in [0, 1]
    private double[] trainY;       // scaled y = (y - floor) / y_scale
    private double[] changepoints; // scaled changepoint times
    private double[][] XSeasonal;  // Fourier features
    private double[][] XHoliday;   // Holiday features
    private double[][] X;          // Combined feature matrix [seasonal | holiday]
    private double[] sigmas;        // Per-component prior scales
    private double[] s_a;           // Additive indicator vector
    private double[] s_m;           // Multiplicative indicator vector
    private double[] capScaled;     // Standardized cap (for logistic)
    private double yScale;          // absmax(y - floor) or (max - min) for minmax
    private double yOffset;         // floor value
    private double tMin, tScale;    // Time scaling params
    private double trainingMaxT;    // Max scaled time in training data

    // Fitted parameters
    private double[] theta;         // [k, m, delta..., logSigma, beta...]
    private double k, m;
    private double[] delta;
    private double sigmaObs;
    private double[] beta;

    // MCMC samples
    private List<double[]> posteriorSamples;

    // Growth type code: 0=linear, 1=logistic, 2=flat
    private int growthCode;

    public ProphetModel(ProphetConfig config) {
        this.config = config;
        this.growthCode = config.growth == ProphetConfig.GrowthType.LINEAR ? 0
                       : config.growth == ProphetConfig.GrowthType.LOGISTIC ? 1 : 2;
    }

    // ===================== Fit =====================

    public void fit(List<DataPoint> data) {
        if (config.verbose) System.out.println("[Prophet] Fitting model with "
            + data.size() + " observations, growth=" + config.growth);

        // --- 1. Setup time scaling ---
        // Reference: forecaster.py setup_dataframe, __init__ self._self.t_scale
        double tMinRaw = Double.MAX_VALUE, tMaxRaw = Double.MIN_VALUE;
        for (DataPoint dp : data) {
            if (dp.getTimestamp() < tMinRaw) tMinRaw = dp.getTimestamp();
            if (dp.getTimestamp() > tMaxRaw) tMaxRaw = dp.getTimestamp();
        }
        this.tMin = tMinRaw;
        this.tScale = tMaxRaw - tMinRaw;
        if (tScale < 1e-10) tScale = 1.0;

        // --- 2. Setup y scaling ---
        // Reference: forecaster.py setup_dataframe, self.y_scale
        // Prophet uses absmax: y_scale = max(abs(y - floor))
        // or minmax: y_scale = max(y-floor) - min(y-floor)
        this.yOffset = config.floor;
        double yMin = Double.MAX_VALUE, yMax = Double.MIN_VALUE;
        double yAbsMax = 0;
        for (DataPoint dp : data) {
            double val = dp.getValue() - yOffset;
            if (val < yMin) yMin = val;
            if (val > yMax) yMax = val;
            if (Math.abs(val) > yAbsMax) yAbsMax = Math.abs(val);
        }
        if (config.scaling == ProphetConfig.Scaling.MINMAX) {
            this.yScale = yMax - yMin;
        } else {
            this.yScale = yAbsMax;
        }
        if (this.yScale < 1e-10) this.yScale = 1.0;

        // --- 3. Scale data ---
        int T = data.size();
        this.trainT = new double[T];
        this.trainY = new double[T];
        for (int i = 0; i < T; i++) {
            DataPoint dp = data.get(i);
            trainT[i] = (dp.getTimestamp() - tMin) / tScale;   // scaled to [0, 1]
            trainY[i] = (dp.getValue() - yOffset) / yScale; // scaled (no mean centering!)
        }
        this.trainingMaxT = trainT[T - 1];

        // --- 4. Set changepoints ---
        // Reference: forecaster.py set_changepoints
        // Prophet: cp_indexes = np.linspace(0, hist_size-1, n+1).round().astype(int)
        //          changepoints = hist[cp_indexes].tail(-1)  (drop first = index 0)
        setChangepoints(T);

        // --- 5. Build seasonal features ---
        // Reference: forecaster.py make_all_seasonality_features
        buildSeasonalFeatures(T);

        // --- 6. Build holiday features ---
        buildHolidayFeatures(T);

        // --- 7. Combine features and set indicators ---
        combineFeatures();

        // --- 8. Standardize cap for logistic ---
        if (growthCode == 1) {
            capScaled = new double[T];
            double capVal = (config.cap - yOffset) / yScale;
            Arrays.fill(capScaled, capVal);
        }

        // --- 9. Initialize trend parameters ---
        // Reference: forecaster.py linear_growth_init / logistic_growth_init / flat_growth_init
        double[] initTheta = initializeTrendParams();

        // --- 10. Optimize (Adam + L-BFGS MAP estimate) ---
        if (config.verbose) System.out.println("[Prophet] Running optimization (Adam + L-BFGS)...");
        optimize(initTheta);

        // --- 11. MCMC if requested ---
        if (config.mcmcSamples > 0) {
            runMCMC();
        } else {
            posteriorSamples = new ArrayList<>();
            posteriorSamples.add(theta.clone());
        }

        if (config.verbose) {
            System.out.printf("[Prophet] Fit complete. k=%.4f m=%.4f sigma_obs=%.4f%n",
                k, m, sigmaObs);
            int significantCP = 0;
            for (double d : delta) if (Math.abs(d) > 1e-6) significantCP++;
            System.out.printf("[Prophet] Changepoints: %d/%d significant, beta dim=%d%n",
                significantCP, delta.length, beta.length);
        }
    }

    // ===================== Changepoints =====================

    private void setChangepoints(int histSize) {
        // Reference: forecaster.py set_changepoints
        // cp_indexes = np.linspace(0, hist_size - 1, n_changepoints + 1).round().astype(int)
        // changepoints = hist_dates[cp_indexes].tail(-1)  # drop the first one
        int n = config.nChangepoints;
        if (n <= 0 || histSize < 2) {
            changepoints = new double[0];
            return;
        }
        // changepoint_range limits where changepoints can be placed
        int histRange = (int) Math.ceil(histSize * config.changepointRange);
        if (histRange < 2) histRange = histSize;

        int[] cpIndexes = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            double idx = (double) i * (histRange - 1) / n;
            cpIndexes[i] = (int) Math.round(idx);
        }

        // Drop the first (index 0), take the rest
        changepoints = new double[n];
        for (int i = 1; i <= n; i++) {
            int dataIdx = Math.min(cpIndexes[i], histSize - 1);
            changepoints[i - 1] = trainT[dataIdx];
        }

        // Sort (should already be sorted)
        Arrays.sort(changepoints);
    }

    // ===================== Seasonal Features =====================

    private void buildSeasonalFeatures(int T) {
        // Reference: forecaster.py fourier_series + make_all_seasonality_features
        // For MIXED mode: generate standard Fourier + time-weighted Fourier features
        // Time-weighted features: t_scaled * sin/cos — amplitude grows with time
        // This captures patterns where seasonal amplitude increases over time
        List<double[]> featureList = new ArrayList<>();
        List<Double> sigmaList = new ArrayList<>();

        boolean isMixed = config.seasonalityMode == ProphetConfig.SeasonalityMode.MIXED;

        if (config.yearlySeasonality) {
            int order = config.yearlyFourierOrder;
            double period = config.yearlyPeriod;
            // Standard Fourier features (constant amplitude)
            for (int k = 1; k <= order; k++) {
                double[] sinCol = new double[T];
                double[] cosCol = new double[T];
                double arg = 2 * Math.PI * k / period;
                for (int i = 0; i < T; i++) {
                    double days = trainT[i] * tScale + tMin;
                    sinCol[i] = Math.sin(days * arg);
                    cosCol[i] = Math.cos(days * arg);
                }
                featureList.add(sinCol);
                featureList.add(cosCol);
                sigmaList.add(config.yearlySeasonalityPriorScale);
                sigmaList.add(config.yearlySeasonalityPriorScale);
            }
            // Time-weighted Fourier features (growing amplitude) — MIXED mode only
            if (isMixed) {
                for (int k = 1; k <= order; k++) {
                    double[] sinCol = new double[T];
                    double[] cosCol = new double[T];
                    double arg = 2 * Math.PI * k / period;
                    for (int i = 0; i < T; i++) {
                        double days = trainT[i] * tScale + tMin;
                        sinCol[i] = trainT[i] * Math.sin(days * arg);  // t_scaled * sin
                        cosCol[i] = trainT[i] * Math.cos(days * arg);  // t_scaled * cos
                    }
                    featureList.add(sinCol);
                    featureList.add(cosCol);
                    sigmaList.add(config.yearlySeasonalityPriorScale);
                    sigmaList.add(config.yearlySeasonalityPriorScale);
                }
            }
        }

        if (config.weeklySeasonality) {
            int order = config.weeklyFourierOrder;
            double period = config.weeklyPeriod;
            for (int k = 1; k <= order; k++) {
                double[] sinCol = new double[T];
                double[] cosCol = new double[T];
                double arg = 2 * Math.PI * k / period;
                for (int i = 0; i < T; i++) {
                    double days = trainT[i] * tScale + tMin;
                    sinCol[i] = Math.sin(days * arg);
                    cosCol[i] = Math.cos(days * arg);
                }
                featureList.add(sinCol);
                featureList.add(cosCol);
                sigmaList.add(config.weeklySeasonalityPriorScale);
                sigmaList.add(config.weeklySeasonalityPriorScale);
            }
            if (isMixed) {
                for (int k = 1; k <= order; k++) {
                    double[] sinCol = new double[T];
                    double[] cosCol = new double[T];
                    double arg = 2 * Math.PI * k / period;
                    for (int i = 0; i < T; i++) {
                        double days = trainT[i] * tScale + tMin;
                        sinCol[i] = trainT[i] * Math.sin(days * arg);
                        cosCol[i] = trainT[i] * Math.cos(days * arg);
                    }
                    featureList.add(sinCol);
                    featureList.add(cosCol);
                    sigmaList.add(config.weeklySeasonalityPriorScale);
                    sigmaList.add(config.weeklySeasonalityPriorScale);
                }
            }
        }

        if (config.dailySeasonality) {
            int order = config.dailyFourierOrder;
            double period = config.dailyPeriod;
            for (int k = 1; k <= order; k++) {
                double[] sinCol = new double[T];
                double[] cosCol = new double[T];
                double arg = 2 * Math.PI * k / period;
                for (int i = 0; i < T; i++) {
                    double days = trainT[i] * tScale + tMin;
                    sinCol[i] = Math.sin(days * arg);
                    cosCol[i] = Math.cos(days * arg);
                }
                featureList.add(sinCol);
                featureList.add(cosCol);
                sigmaList.add(config.dailySeasonalityPriorScale);
                sigmaList.add(config.dailySeasonalityPriorScale);
            }
            if (isMixed) {
                for (int k = 1; k <= order; k++) {
                    double[] sinCol = new double[T];
                    double[] cosCol = new double[T];
                    double arg = 2 * Math.PI * k / period;
                    for (int i = 0; i < T; i++) {
                        double days = trainT[i] * tScale + tMin;
                        sinCol[i] = trainT[i] * Math.sin(days * arg);
                        cosCol[i] = trainT[i] * Math.cos(days * arg);
                    }
                    featureList.add(sinCol);
                    featureList.add(cosCol);
                    sigmaList.add(config.dailySeasonalityPriorScale);
                    sigmaList.add(config.dailySeasonalityPriorScale);
                }
            }
        }

        // Convert list to 2D array
        int K = featureList.size();
        if (K > 0) {
            XSeasonal = new double[T][K];
            for (int j = 0; j < K; j++) {
                double[] col = featureList.get(j);
                for (int i = 0; i < T; i++) {
                    XSeasonal[i][j] = col[i];
                }
            }
        } else {
            XSeasonal = null;
        }

        // Store seasonal sigmas
        seasonalSigmas = sigmaList.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private double[] seasonalSigmas;

    // ===================== Holiday Features =====================

    private void buildHolidayFeatures(int T) {
        if (config.holidays.isEmpty()) {
            XHoliday = null;
            return;
        }

        // Reference: forecaster.py make_holiday_features
        // Each holiday generates columns for [window_before, day_of, window_after]
        List<double[]> featureList = new ArrayList<>();
        List<Double> sigmaList = new ArrayList<>();

        for (ProphetConfig.HolidaySpec hol : config.holidays) {
            double holDay = hol.timestamp;
            int wb = hol.windowBefore;
            int wa = hol.windowAfter;

            // Window days: -wb, ..., 0, ..., +wa
            for (int offset = -wb; offset <= wa; offset++) {
                double[] col = new double[T];
                double targetDay = holDay + offset;
                for (int i = 0; i < T; i++) {
                    double days = trainT[i] * tScale + tMin;
                    col[i] = Math.abs(days - targetDay) < 0.5 ? 1.0 : 0.0;
                }
                featureList.add(col);
                sigmaList.add(config.holidaysPriorScale);
            }
        }

        int K = featureList.size();
        XHoliday = new double[T][K];
        for (int j = 0; j < K; j++) {
            double[] col = featureList.get(j);
            for (int i = 0; i < T; i++) {
                XHoliday[i][j] = col[i];
            }
        }
        holidaySigmas = sigmaList.stream().mapToDouble(Double::doubleValue).toArray();
    }

    private double[] holidaySigmas;

    // ===================== Combine Features =====================

    private void combineFeatures() {
        int Ks = XSeasonal != null ? XSeasonal[0].length : 0;
        int Kh = XHoliday != null ? XHoliday[0].length : 0;
        int K = Ks + Kh;
        int T = trainT.length;

        if (K == 0) {
            X = null;
            sigmas = new double[0];
            s_a = new double[0];
            s_m = new double[0];
            return;
        }

        X = new double[T][K];
        sigmas = new double[K];
        s_a = new double[K];
        s_m = new double[K];

        boolean isAdditive = config.seasonalityMode == ProphetConfig.SeasonalityMode.ADDITIVE;
        boolean isMultiplicative = config.seasonalityMode == ProphetConfig.SeasonalityMode.MULTIPLICATIVE;
        boolean isMixed = config.seasonalityMode == ProphetConfig.SeasonalityMode.MIXED;
        boolean holidaysAdditive = config.getHolidaysMode() == ProphetConfig.SeasonalityMode.ADDITIVE;

        // For MIXED mode: ALL features are additive (s_a=1).
        // Standard Fourier + time-weighted Fourier both contribute directly.
        // Time-weighted features (t_scaled * sin/cos) naturally have growing amplitude.

        for (int i = 0; i < T; i++) {
            int col = 0;
            // Seasonal features
            for (int j = 0; j < Ks; j++) {
                X[i][col] = XSeasonal[i][j];
                sigmas[col] = seasonalSigmas[j];
                if (isAdditive || isMixed) {
                    s_a[col] = 1.0; s_m[col] = 0.0;
                } else { // MULTIPLICATIVE
                    s_a[col] = 0.0; s_m[col] = 1.0;
                }
                col++;
            }
            // Holiday features
            for (int j = 0; j < Kh; j++) {
                X[i][col] = XHoliday[i][j];
                sigmas[col] = holidaySigmas[j];
                s_a[col] = holidaysAdditive ? 1.0 : 0.0;
                s_m[col] = holidaysAdditive ? 0.0 : 1.0;
                col++;
            }
        }
    }

    // ===================== Trend Initialization =====================

    /**
     * Initialize trend parameters using Prophet's growth_init methods,
     * then use least-squares on residuals to initialize beta (seasonal coefficients).
     * This provides a much better starting point for L-BFGS optimization.
     */
    private double[] initializeTrendParams() {
        int S = changepoints.length;
        int K = X != null ? X[0].length : 0;
        int nParams = 3 + S + K;

        double[] init = new double[nParams];

        // --- Phase 1: Initialize k, m using Prophet's growth_init ---
        if (growthCode == 2) {
            double sum = 0;
            for (double v : trainY) sum += v;
            init[1] = sum / trainY.length;  // m
            init[0] = 0;  // k (unused for flat)
        } else if (growthCode == 1) {
            double capVal = (config.cap - yOffset) / yScale;
            double[] yCapped = new double[trainY.length];
            for (int i = 0; i < trainY.length; i++) {
                yCapped[i] = Math.min(capVal, Math.max(0.01, trainY[i]));
            }
            double denom0 = Math.max(capVal - yCapped[0], 1e-8);
            double denom1 = Math.max(capVal - yCapped[trainY.length - 1], 1e-8);
            double r0 = Math.log(Math.max(yCapped[0] / denom0, 1e-8));
            double r1 = Math.log(Math.max(yCapped[trainY.length - 1] / denom1, 1e-8));
            double lr = r1 - r0;
            if (Math.abs(lr) < 1e-8) lr = 1e-8;
            double k0 = (yCapped[trainY.length - 1] - yCapped[0]) / lr;
            double m0 = yCapped[0] - k0 * r0;
            init[0] = k0;
            init[1] = m0;
        } else {
            // Linear growth init: k = (y[-1] - y[0]) / (t[-1] - t[0])
            double t0 = trainT[0];
            double tN = trainT[trainT.length - 1];
            double y0 = trainY[0];
            double yN = trainY[trainY.length - 1];
            double dt = tN - t0;
            if (Math.abs(dt) < 1e-10) dt = 1.0;
            init[0] = (yN - y0) / dt;  // k
            init[1] = y0 - init[0] * t0;  // m
        }

        // delta = 0 (all changepoints start at 0)
        // logSigma = log(noise_std) - estimate from residuals
        if (K > 0) {
            // --- Phase 2: Least-squares initialization for beta ---
            // Compute trend prediction with current k, m, delta=0
            double[] trendInit = new double[trainT.length];
            for (int i = 0; i < trainT.length; i++) {
                trendInit[i] = init[0] * trainT[i] + init[1];  // linear: k*t + m
            }

            // Residuals = y - trend
            double[] residuals = new double[trainT.length];
            double resSumSq = 0;
            for (int i = 0; i < trainT.length; i++) {
                residuals[i] = trainY[i] - trendInit[i];
                resSumSq += residuals[i] * residuals[i];
            }
            double noiseStd = Math.sqrt(resSumSq / trainT.length);

            // Use ridge regression to fit beta on residuals
            // For additive: residuals ≈ X * beta
            // For multiplicative: residuals ≈ trend * X * beta
            // We solve a weighted least squares depending on mode
            double[][] Xls;
            if (config.seasonalityMode == ProphetConfig.SeasonalityMode.MULTIPLICATIVE) {
                // multiplicative: y - trend ≈ trend * (X * beta)
                // so (y - trend) / trend ≈ X * beta (weighted)
                Xls = new double[trainT.length][K];
                for (int i = 0; i < trainT.length; i++) {
                    double w = Math.abs(trendInit[i]) + 1e-8;
                    for (int j = 0; j < K; j++) {
                        Xls[i][j] = X[i][j] / w;
                    }
                    residuals[i] /= w;
                }
            } else {
                Xls = X;
            }

            // Solve ridge regression: (X'X + lambda*I) beta = X'r
            // lambda = small regularization to avoid overfitting
            double lambda = 1e-3;
            try {
                Matrix Xmat = new Matrix(Xls);
                double[] betaInit = Matrix.solveLeastSquares(Xmat, residuals, lambda);
                // Clip beta to reasonable range
                for (int j = 0; j < K; j++) {
                    init[3 + S + j] = Math.max(-10, Math.min(10, betaInit[j]));
                }
            } catch (Exception e) {
                // If LS fails, leave beta at 0
            }

            init[2 + S] = Math.log(Math.max(noiseStd, 1e-4));
        } else {
            // No seasonal features: estimate sigma from data
            double resSumSq = 0;
            for (int i = 0; i < trainY.length; i++) {
                double pred = init[0] * trainT[i] + init[1];
                resSumSq += (trainY[i] - pred) * (trainY[i] - pred);
            }
            double noiseStd = Math.sqrt(resSumSq / trainT.length);
            init[2 + S] = Math.log(Math.max(noiseStd, 1e-4));
        }

        return init;
    }

    // ===================== Optimization =====================

    private void optimize(double[] initTheta) {
        AdamOptimizer.ObjectiveFunction fn = (theta) -> BayesianPriors.negLogPosteriorAndGradient(
                theta, trainT, trainY, changepoints, X, sigmas, s_a, s_m,
                config.changepointPriorScale, config.sigmaObsPriorScale,
                growthCode, capScaled
        );

        // Phase 1: Adam optimizer (handles Laplace priors well)
        AdamOptimizer adam = new AdamOptimizer(5000, 0.005, 0.9, 0.999, 1e-8, 1e-4, config.verbose);
        AdamOptimizer.OptResult adamResult = adam.minimize(initTheta, fn);

        theta = adamResult.params;

        if (config.verbose) {
            System.out.printf("  Adam: f=%.6f, |grad|=%.4e%n", adamResult.value, adamResult.gradNorm);
        }

        // Extract fitted parameters
        extractParams(theta);
    }

    private void extractParams(double[] theta) {
        int S = changepoints.length;
        int K = X != null ? X[0].length : 0;
        this.theta = theta;
        this.k = theta[0];
        this.m = theta[1];
        this.delta = new double[S];
        System.arraycopy(theta, 2, delta, 0, S);
        this.sigmaObs = Math.exp(theta[2 + S]);
        this.beta = new double[K];
        if (K > 0) System.arraycopy(theta, 3 + S, beta, 0, K);
    }

    // ===================== MCMC =====================

    private void runMCMC() {
        if (config.verbose) System.out.println("[Prophet] Running MCMC: "
            + config.mcmcChains + " chains x " + config.mcmcSamples + " samples");

        MCMCSampler.LogPosteriorFn logPostFn = (theta) -> {
            double[] result = BayesianPriors.negLogPosteriorAndGradient(
                    theta, trainT, trainY, changepoints, X, sigmas, s_a, s_m,
                    config.changepointPriorScale, config.sigmaObsPriorScale,
                    growthCode, capScaled
            );
            return -result[0];  // return logPosterior (negate negLogPost)
        };

        int samplesPerChain = Math.max(1, config.mcmcSamples / config.mcmcChains);
        MCMCSampler sampler = new MCMCSampler(
                samplesPerChain, config.mcmcWarmup, config.mcmcChains,
                config.useNUTS, config.verbose
        );

        double[][] samples = sampler.sample(theta, logPostFn);

        posteriorSamples = new ArrayList<>();
        for (double[] s : samples) {
            posteriorSamples.add(s.clone());
        }

        if (config.verbose) {
            System.out.println("[Prophet] MCMC complete: " + posteriorSamples.size() + " posterior samples");
        }
    }

    // ===================== Predict =====================

    public static class ForecastResult {
        public double[] t;        // original time values
        public double[] yhat;     // point predictions (original scale)
        public double[] yhatLower;  // lower bound (original scale)
        public double[] yhatUpper;  // upper bound (original scale)
        public double[] trend;    // trend component (original scale)
        public double[] seasonal; // seasonal component (original scale)

        public ForecastResult(int n) {
            t = new double[n];
            yhat = new double[n];
            yhatLower = new double[n];
            yhatUpper = new double[n];
            trend = new double[n];
            seasonal = new double[n];
        }
    }

    public ForecastResult predict(double[] futureT) {
        return predict(futureT, config.uncertaintySamples);
    }

    public ForecastResult predict(double[] futureT, int nUncertaintySamples) {
        int N = futureT.length;

        // Scale future time
        double[] tScaled = new double[N];
        for (int i = 0; i < N; i++) {
            tScaled[i] = (futureT[i] - tMin) / tScale;
        }

        // Scale cap for logistic
        double[] futureCapScaled = null;
        if (growthCode == 1) {
            futureCapScaled = new double[N];
            double capVal = (config.cap - yOffset) / yScale;
            Arrays.fill(futureCapScaled, capVal);
        }

        // Build future features
        double[][] XFuture = buildFutureFeatures(futureT, N);

        // Point prediction using MAP estimate
        double[] trendArr = BayesianPriors.computeTrend(k, m, delta, tScaled, changepoints, growthCode, futureCapScaled);

        double[] yhatScaled = new double[N];
        double[] seasonalArr = new double[N];
        for (int i = 0; i < N; i++) {
            double multPart = 0, addPart = 0;
            if (XFuture != null) {
                for (int j = 0; j < XFuture[i].length; j++) {
                    multPart += XFuture[i][j] * s_m[j] * beta[j];
                    addPart += XFuture[i][j] * s_a[j] * beta[j];
                }
            }
            yhatScaled[i] = trendArr[i] * (1.0 + multPart) + addPart;
            seasonalArr[i] = multPart * trendArr[i] + addPart;  // seasonal + holiday contribution
        }

        // Convert to original scale
        ForecastResult result = new ForecastResult(N);
        for (int i = 0; i < N; i++) {
            result.t[i] = futureT[i];
            result.yhat[i] = yhatScaled[i] * yScale + yOffset;
            result.trend[i] = trendArr[i] * yScale + yOffset;
            result.seasonal[i] = seasonalArr[i] * yScale;
        }

        // Uncertainty estimation
        if (nUncertaintySamples > 0 && !posteriorSamples.isEmpty()) {
            computeUncertainty(tScaled, XFuture, futureCapScaled, N, nUncertaintySamples, result);
        } else if (nUncertaintySamples > 0) {
            // No MCMC samples: use MAP estimate with sigma_obs-based intervals
            // plus trend uncertainty from Poisson future changepoints
            computeMapUncertainty(tScaled, XFuture, futureCapScaled, N, nUncertaintySamples, result);
        } else {
            for (int i = 0; i < N; i++) {
                result.yhatLower[i] = result.yhat[i];
                result.yhatUpper[i] = result.yhat[i];
            }
        }

        return result;
    }

    // ===================== Uncertainty =====================

    /**
     * Compute prediction uncertainty using posterior samples.
     *
     * Reference: forecaster.py predict_uncertainty, predictive_samples,
     *            sample_predictive_trend
     *
     * For future time points (t > trainingMaxT), Prophet generates new changepoints
     * using a Poisson process:
     *   n_changes ~ Poisson(S * (T_future - 1))
     *   new changepoints uniformly distributed in [1, T_future]
     *   new deltas ~ Laplace(0, mean(|training_deltas|) + 1e-8)
     */

    /**
     * Compute prediction uncertainty using MAP estimate only (no MCMC).
     * Uses sigma_obs for observation noise and Poisson future changepoints for trend uncertainty.
     */
    private void computeMapUncertainty(
            double[] tScaled, double[][] XFuture, double[] futureCapScaled,
            int N, int nSamples, ForecastResult result) {

        // Compute mean absolute delta for Poisson trend uncertainty
        double meanAbsDelta = 0;
        for (double d : delta) meanAbsDelta += Math.abs(d);
        meanAbsDelta /= delta.length;
        double deltaScale = meanAbsDelta + 1e-8;
        int S = changepoints.length;

        Random rng = new Random(42);
        double[][] yhatSamples = new double[nSamples][N];

        for (int s = 0; s < nSamples; s++) {
            // Generate future changepoints via Poisson process
            double[] sampleDelta = delta.clone();
            if (S > 0) {
                // For future time points beyond training range
                double futureRange = tScaled[N - 1] - trainingMaxT;
                if (futureRange > 0) {
                    int nNewCps = (int) Math.round(S * futureRange);
                    for (int nc = 0; nc < nNewCps; nc++) {
                        // Random new delta with Laplace distribution
                        double u = rng.nextDouble() - 0.5;
                        double newDelta = -deltaScale * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
                        // This is added to the cumulative rate
                        // We approximate by adding to the last delta
                        if (sampleDelta.length > 0) {
                            sampleDelta[sampleDelta.length - 1] += newDelta * 0.5;
                        }
                    }
                }
            }

            // Compute trend with perturbed deltas
            double[] trendSample = BayesianPriors.computeTrend(
                k, m, sampleDelta, tScaled, changepoints, growthCode, futureCapScaled);

            // Compute seasonal (same as MAP since beta is fixed)
            for (int i = 0; i < N; i++) {
                double multPart = 0, addPart = 0;
                if (XFuture != null) {
                    for (int j = 0; j < XFuture[i].length; j++) {
                        multPart += XFuture[i][j] * s_m[j] * beta[j];
                        addPart += XFuture[i][j] * s_a[j] * beta[j];
                    }
                }
                double yhatS = trendSample[i] * (1.0 + multPart) + addPart;
                // Add observation noise
                yhatS += rng.nextGaussian() * sigmaObs;
                yhatSamples[s][i] = yhatS * yScale + yOffset;
            }
        }

        // Compute 80% CI from samples (10th and 90th percentile)
        for (int i = 0; i < N; i++) {
            double[] vals = new double[nSamples];
            for (int s = 0; s < nSamples; s++) vals[s] = yhatSamples[s][i];
            Arrays.sort(vals);
            int lo10 = (int) Math.round(0.10 * nSamples);
            int hi90 = (int) Math.round(0.90 * nSamples);
            lo10 = Math.max(0, Math.min(lo10, nSamples - 1));
            hi90 = Math.max(0, Math.min(hi90, nSamples - 1));
            result.yhatLower[i] = vals[lo10];
            result.yhatUpper[i] = vals[hi90];
        }
    }

    private void computeUncertainty(
            double[] tScaled, double[][] XFuture, double[] futureCapScaled,
            int N, int nSamples, ForecastResult result) {

        // Determine number of historical vs future points
        // Future points have t > trainingMaxT
        Random rng = new Random(42);

        // Compute mean |delta| for new changepoint generation
        double meanAbsDelta = 0;
        for (double d : delta) meanAbsDelta += Math.abs(d);
        meanAbsDelta = meanAbsDelta / delta.length + 1e-8;

        // Simulate nSamples posterior draws
        double[][] simY = new double[nSamples][N];

        for (int s = 0; s < nSamples; s++) {
            // Sample from posterior (cycle through posterior samples)
            double[] sample = posteriorSamples.get(s % posteriorSamples.size());
            int S = changepoints.length;
            int K = X != null ? X[0].length : 0;

            double sK = sample[0];
            double sM = sample[1];
            double[] sDelta = new double[S];
            System.arraycopy(sample, 2, sDelta, 0, S);
            double sSigma = Math.exp(sample[2 + S]);
            double[] sBeta = new double[K];
            if (K > 0) System.arraycopy(sample, 3 + S, sBeta, 0, K);

            // Generate trend with future changepoints (Poisson process)
            double[] trendSim = samplePredictiveTrend(
                    sK, sM, sDelta, tScaled, futureCapScaled, N, rng, meanAbsDelta
            );

            // Add seasonal + holiday + noise
            for (int i = 0; i < N; i++) {
                double multPart = 0, addPart = 0;
                if (XFuture != null) {
                    for (int j = 0; j < XFuture[i].length; j++) {
                        multPart += XFuture[i][j] * s_m[j] * sBeta[j];
                        addPart += XFuture[i][j] * s_a[j] * sBeta[j];
                    }
                }
                double yhat = trendSim[i] * (1.0 + multPart) + addPart;
                // Add observation noise
                simY[s][i] = yhat + rng.nextGaussian() * sSigma;
            }
        }

        // Compute percentiles
        for (int i = 0; i < N; i++) {
            double[] vals = new double[nSamples];
            for (int s = 0; s < nSamples; s++) {
                vals[s] = simY[s][i] * yScale + yOffset;
            }
            Arrays.sort(vals);
            int lowerIdx = (int) Math.floor(0.1 * nSamples);
            int upperIdx = (int) Math.ceil(0.9 * nSamples);
            if (lowerIdx >= nSamples) lowerIdx = nSamples - 1;
            if (upperIdx >= nSamples) upperIdx = nSamples - 1;
            result.yhatLower[i] = vals[lowerIdx];
            result.yhatUpper[i] = vals[upperIdx];
        }
    }

    /**
     * Sample predictive trend with Poisson-generated future changepoints.
     *
     * Reference: forecaster.py sample_predictive_trend
     *
     * For points beyond training data, generate new changepoints:
     *   n_changes ~ Poisson(S * (T - 1))  where T is the future time horizon
     *   new changepoints ~ Uniform[1, T]
     *   new deltas ~ Laplace(0, mean(|training_deltas|))
     */
    private double[] samplePredictiveTrend(
            double sK, double sM, double[] sDelta,
            double[] tScaled, double[] futureCapScaled,
            int N, Random rng, double meanAbsDelta) {

        int S = sDelta.length;

        // Determine if we have future points (t > trainingMaxT)
        boolean hasFuture = false;
        for (int i = 0; i < N; i++) {
            if (tScaled[i] > trainingMaxT + 1e-10) {
                hasFuture = true;
                break;
            }
        }

        if (!hasFuture || S == 0) {
            // No future changepoints needed
            return BayesianPriors.computeTrend(sK, sM, sDelta, tScaled, changepoints, growthCode, futureCapScaled);
        }

        // Generate future changepoints via Poisson process
        double lambda = S * (tScaled[N - 1] - trainingMaxT) / Math.max(trainingMaxT, 1e-10);
        // Cap lambda to avoid excessive changepoints
        lambda = Math.min(lambda, 50);
        int nFutureChanges = rng.nextDouble() < lambda ?
                (int) Math.ceil(lambda) : (int) Math.floor(lambda);

        if (nFutureChanges <= 0) {
            return BayesianPriors.computeTrend(sK, sM, sDelta, tScaled, changepoints, growthCode, futureCapScaled);
        }

        // Generate future changepoint times and deltas
        double[] futureCPs = new double[nFutureChanges];
        double[] futureDeltas = new double[nFutureChanges];
        for (int i = 0; i < nFutureChanges; i++) {
            futureCPs[i] = trainingMaxT + rng.nextDouble() * (tScaled[N - 1] - trainingMaxT);
            // delta ~ Laplace(0, meanAbsDelta)
            double u = rng.nextDouble() - 0.5;
            futureDeltas[i] = -2 * meanAbsDelta * Math.signum(u) * Math.log(1 - 2 * Math.abs(u));
        }
        Arrays.sort(futureCPs);

        // Combine training + future changepoints
        double[] allCPs = new double[S + nFutureChanges];
        double[] allDeltas = new double[S + nFutureChanges];
        System.arraycopy(changepoints, 0, allCPs, 0, S);
        System.arraycopy(sDelta, 0, allDeltas, 0, S);
        System.arraycopy(futureCPs, 0, allCPs, S, nFutureChanges);
        System.arraycopy(futureDeltas, 0, allDeltas, S, nFutureChanges);

        // Sort by changepoint time
        // (changepoints should already be sorted, future ones are sorted)
        // Merge sort the combined arrays
        double[] sortedCPs = new double[S + nFutureChanges];
        double[] sortedDeltas = new double[S + nFutureChanges];
        int i1 = 0, i2 = 0;
        for (int i = 0; i < S + nFutureChanges; i++) {
            if (i2 >= nFutureChanges || (i1 < S && changepoints[i1] <= futureCPs[i2])) {
                sortedCPs[i] = changepoints[i1];
                sortedDeltas[i] = sDelta[i1];
                i1++;
            } else {
                sortedCPs[i] = futureCPs[i2];
                sortedDeltas[i] = futureDeltas[i2];
                i2++;
            }
        }

        return BayesianPriors.computeTrend(sK, sM, sortedDeltas, tScaled, sortedCPs, growthCode, futureCapScaled);
    }

    // ===================== Feature Building for Future =====================

    private double[][] buildFutureFeatures(double[] futureT, int N) {
        if (X == null) return null;

        int K = X[0].length;
        double[][] XFuture = new double[N][K];

        boolean isMixed = config.seasonalityMode == ProphetConfig.SeasonalityMode.MIXED;

        for (int i = 0; i < N; i++) {
            double days = futureT[i];
            // Compute scaled time for MIXED mode's time-weighted features
            double tScaled = (days - tMin) / tScale;
            int col = 0;

            // Seasonal features (same structure as buildSeasonalFeatures)
            if (config.yearlySeasonality) {
                int order = config.yearlyFourierOrder;
                double period = config.yearlyPeriod;
                // Standard yearly features
                for (int k = 1; k <= order; k++) {
                    double arg = 2 * Math.PI * k / period;
                    XFuture[i][col++] = Math.sin(days * arg);
                    XFuture[i][col++] = Math.cos(days * arg);
                }
                // Time-weighted yearly features (MIXED mode only)
                if (isMixed) {
                    for (int k = 1; k <= order; k++) {
                        double arg = 2 * Math.PI * k / period;
                        XFuture[i][col++] = tScaled * Math.sin(days * arg);
                        XFuture[i][col++] = tScaled * Math.cos(days * arg);
                    }
                }
            }
            if (config.weeklySeasonality) {
                int order = config.weeklyFourierOrder;
                double period = config.weeklyPeriod;
                for (int k = 1; k <= order; k++) {
                    double arg = 2 * Math.PI * k / period;
                    XFuture[i][col++] = Math.sin(days * arg);
                    XFuture[i][col++] = Math.cos(days * arg);
                }
                if (isMixed) {
                    for (int k = 1; k <= order; k++) {
                        double arg = 2 * Math.PI * k / period;
                        XFuture[i][col++] = tScaled * Math.sin(days * arg);
                        XFuture[i][col++] = tScaled * Math.cos(days * arg);
                    }
                }
            }
            if (config.dailySeasonality) {
                int order = config.dailyFourierOrder;
                double period = config.dailyPeriod;
                for (int k = 1; k <= order; k++) {
                    double arg = 2 * Math.PI * k / period;
                    XFuture[i][col++] = Math.sin(days * arg);
                    XFuture[i][col++] = Math.cos(days * arg);
                }
                if (isMixed) {
                    for (int k = 1; k <= order; k++) {
                        double arg = 2 * Math.PI * k / period;
                        XFuture[i][col++] = tScaled * Math.sin(days * arg);
                        XFuture[i][col++] = tScaled * Math.cos(days * arg);
                    }
                }
            }

            // Holiday features
            if (XHoliday != null) {
                for (ProphetConfig.HolidaySpec hol : config.holidays) {
                    int wb = hol.windowBefore;
                    int wa = hol.windowAfter;
                    for (int offset = -wb; offset <= wa; offset++) {
                        double targetDay = hol.timestamp + offset;
                        XFuture[i][col++] = Math.abs(days - targetDay) < 0.5 ? 1.0 : 0.0;
                    }
                }
            }
        }

        return XFuture;
    }

    // ===================== Getters =====================

    public ProphetConfig getConfig() { return config; }
    public double getSigmaObs() { return sigmaObs * yScale; }  // back to original scale
    public double getSigmaObsRaw() { return sigmaObs; }        // scaled
    public double[] getDelta() { return delta; }
    public double[] getBeta() { return beta; }
    public double getK() { return k; }
    public double getM() { return m; }
    public double getYScale() { return yScale; }
    public double getYOffset() { return yOffset; }
    public double getTMin() { return tMin; }
    public double getTScale() { return tScale; }
    public double getTrainingMaxT() { return trainingMaxT; }
    public int getGrowthCode() { return growthCode; }
    public double[] getChangepoints() { return changepoints; }
    public double[] getSA() { return s_a; }
    public double[] getSM() { return s_m; }
    public double[] getSigmas() { return sigmas; }
    public double[] getCapScaled() { return capScaled; }
    public double[] getTrainT() { return trainT; }
    public double[] getTrainY() { return trainY; }
    public List<double[]> getPosteriorSamples() { return posteriorSamples; }

    // ===================== Setters (for deserialization) =====================

    public void setK(double k) { this.k = k; }
    public void setM(double m) { this.m = m; }
    public void setSigmaObs(double s) { this.sigmaObs = s; }
    public void setDelta(double[] d) { this.delta = d; }
    public void setBeta(double[] b) { this.beta = b; }
}
