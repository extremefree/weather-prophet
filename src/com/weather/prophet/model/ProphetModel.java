package com.weather.prophet.model;

import com.weather.prophet.data.DataPoint;

import java.util.ArrayList;
import java.util.List;

/**
 * ProphetModel - A Java implementation inspired by Facebook's Prophet.
 *
 * Core model: y(t) = g(t) + s(t) + h(t) + ε(t)
 *
 * Where:
 *   g(t) = piecewise linear trend (TrendComponent)
 *   s(t) = Fourier series seasonality (SeasonalityComponent)
 *   h(t) = holiday/special event effects (HolidayComponent)
 *   ε(t) = normally distributed error
 *
 * The model decomposes a time series into trend, seasonality, and holiday
 * components, each fitted independently via least-squares regression.
 *
 * Usage:
 *   ProphetModel model = new ProphetModel();
 *   model.addSeasonality("yearly", 365.25, 10);
 *   model.addSeasonality("weekly", 7, 3);
 *   model.fit(data);
 *   double[] forecast = model.predict(futureTimestamps);
 */
public class ProphetModel {

    // Model components
    private final TrendComponent trend;
    private final List<SeasonalityComponent> seasonalities;
    private HolidayComponent holidays;

    // Fitted state
    private boolean isFitted = false;
    private double[] residuals;
    private double meanError;
    private double stdError;

    // Configuration
    private int numChangepoints = 25;

    public ProphetModel() {
        this.trend = new TrendComponent(numChangepoints);
        this.seasonalities = new ArrayList<>();
    }

    public ProphetModel(int numChangepoints) {
        this.numChangepoints = numChangepoints;
        this.trend = new TrendComponent(numChangepoints);
        this.seasonalities = new ArrayList<>();
    }

    /**
     * Add a seasonality component.
     *
     * @param name         descriptive name ("yearly", "weekly", "daily")
     * @param period       period in days
     * @param fourierOrder number of Fourier harmonics
     * @return this (for chaining)
     */
    public ProphetModel addSeasonality(String name, double period, int fourierOrder) {
        seasonalities.add(new SeasonalityComponent(name, period, fourierOrder));
        return this;
    }

    /**
     * Set holiday component.
     */
    public ProphetModel setHolidays(HolidayComponent holidays) {
        this.holidays = holidays;
        return this;
    }

    /**
     * Fit the model to historical data.
     *
     * Fitting procedure (following Prophet's approach):
     * 1. Fit the trend component
     * 2. Compute residuals (data - trend)
     * 3. Fit each seasonality component to the residuals
     * 4. Compute remaining residuals for error estimation
     * 5. Fit holiday effects (if provided)
     *
     * @param data list of DataPoints (must be sorted by timestamp)
     */
    public void fit(List<DataPoint> data) {
        System.out.println("[ProphetModel] Fitting model on " + data.size() + " data points...");

        int n = data.size();

        // Step 1: Fit trend
        System.out.println("[ProphetModel] Step 1: Fitting trend component...");
        trend.fit(data);
        double[] trendValues = trend.predict(extractTimestamps(data));

        // Step 2: Fit each seasonality component sequentially
        double[] currentResiduals = new double[n];
        for (int i = 0; i < n; i++) {
            currentResiduals[i] = data.get(i).getValue() - trendValues[i];
        }

        for (SeasonalityComponent season : seasonalities) {
            System.out.println("[ProphetModel] Fitting seasonality: " + season.getName() +
                    " (period=" + season.getPeriod() + ")");
            // Fit against current residuals (trend + previously fitted seasonalities removed)
            double[] fittedSoFar = new double[n];
            for (int i = 0; i < n; i++) {
                fittedSoFar[i] = trendValues[i];
                // Subtract already-fitted seasonalities
                for (SeasonalityComponent fitted : seasonalities) {
                    if (fitted == season) break;
                    if (fitted.getCoefficients() != null) {
                        double[] sv = fitted.predict(extractTimestamps(data));
                        fittedSoFar[i] += sv[i];
                    }
                }
            }
            season.fit(data, fittedSoFar);
        }

        // Step 3: Fit holidays (if provided)
        if (holidays != null) {
            System.out.println("[ProphetModel] Fitting holiday component...");
            // Compute fitted values from trend + all seasonalities
            double[] ts = extractTimestamps(data);
            double[] trendV = trend.predict(ts);
            double[] seasonalityV = new double[n];
            for (SeasonalityComponent season : seasonalities) {
                if (season.getCoefficients() != null) {
                    double[] sv = season.predict(ts);
                    for (int i = 0; i < n; i++) seasonalityV[i] += sv[i];
                }
            }
            // Fit holidays against residuals (data - trend - seasonality)
            holidays.fit(data, trendV, seasonalityV);
        }

        // Step 4: Compute final residuals and error statistics
        double[] predicted = computeTotalFitted(data);
        residuals = new double[n];
        double sumError = 0;
        for (int i = 0; i < n; i++) {
            residuals[i] = data.get(i).getValue() - predicted[i];
            sumError += residuals[i];
        }
        meanError = sumError / n;

        double sumSqError = 0;
        for (int i = 0; i < n; i++) {
            sumSqError += (residuals[i] - meanError) * (residuals[i] - meanError);
        }
        stdError = Math.sqrt(sumSqError / n);

        isFitted = true;

        // Print fit summary
        double mae = 0;
        double rmse = 0;
        for (int i = 0; i < n; i++) {
            mae += Math.abs(residuals[i]);
            rmse += residuals[i] * residuals[i];
        }
        mae /= n;
        rmse = Math.sqrt(rmse / n);

        System.out.println("[ProphetModel] Fit complete!");
        System.out.printf("[ProphetModel] Training metrics - MAE: %.4f, RMSE: %.4f, StdErr: %.4f%n",
                mae, rmse, stdError);
    }

    /**
     * Predict future values.
     *
     * @param timestamps array of future timestamps
     * @return predicted values
     */
    public double[] predict(double[] timestamps) {
        if (!isFitted) throw new IllegalStateException("Model must be fitted before prediction");

        int n = timestamps.length;
        double[] result = new double[n];

        // Sum all components
        double[] trendPred = trend.predict(timestamps);
        for (int i = 0; i < n; i++) {
            result[i] = trendPred[i];
        }

        for (SeasonalityComponent season : seasonalities) {
            double[] seasonPred = season.predict(timestamps);
            for (int i = 0; i < n; i++) {
                result[i] += seasonPred[i];
            }
        }

        if (holidays != null) {
            double[] holidayPred = holidays.predict(timestamps);
            for (int i = 0; i < n; i++) {
                result[i] += holidayPred[i];
            }
        }

        return result;
    }

    /**
     * Generate forecast with confidence intervals.
     *
     * @param timestamps future timestamps
     * @param confidence confidence level (e.g., 0.8 for 80% interval)
     * @return ForecastResult with predictions and intervals
     */
    public ForecastResult predict(double[] timestamps, double confidence) {
        double[] yhat = predict(timestamps);
        int n = yhat.length;

        // Z-score for confidence interval
        double zScore = getZScore(confidence);

        double[] yhatLower = new double[n];
        double[] yhatUpper = new double[n];

        for (int i = 0; i < n; i++) {
            yhatLower[i] = yhat[i] - zScore * stdError;
            yhatUpper[i] = yhat[i] + zScore * stdError;
        }

        return new ForecastResult(timestamps, yhat, yhatLower, yhatUpper);
    }

    /**
     * Compute the total fitted values for training data.
     */
    private double[] computeTotalFitted(List<DataPoint> data) {
        double[] timestamps = extractTimestamps(data);
        int n = data.size();
        double[] result = new double[n];

        double[] trendValues = trend.predict(timestamps);
        for (int i = 0; i < n; i++) result[i] += trendValues[i];

        for (SeasonalityComponent season : seasonalities) {
            if (season.getCoefficients() != null) {
                double[] sv = season.predict(timestamps);
                for (int i = 0; i < n; i++) result[i] += sv[i];
            }
        }

        if (holidays != null) {
            double[] hv = holidays.predict(timestamps);
            for (int i = 0; i < n; i++) result[i] += hv[i];
        }

        return result;
    }

    private double[] extractTimestamps(List<DataPoint> data) {
        double[] timestamps = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            timestamps[i] = data.get(i).getTimestamp();
        }
        return timestamps;
    }

    /**
     * Approximate z-score for confidence interval.
     */
    private double getZScore(double confidence) {
        // Approximation of inverse normal CDF
        if (confidence >= 0.99) return 2.576;
        if (confidence >= 0.95) return 1.960;
        if (confidence >= 0.90) return 1.645;
        if (confidence >= 0.80) return 1.282;
        return 1.0;
    }

    // Getters
    public TrendComponent getTrend() { return trend; }
    public List<SeasonalityComponent> getSeasonalities() { return seasonalities; }
    public HolidayComponent getHolidays() { return holidays; }
    public double getStdError() { return stdError; }
    public boolean isFitted() { return isFitted; }

    /**
     * Forecast result container with point predictions and confidence intervals.
     */
    public static class ForecastResult {
        public final double[] timestamps;
        public final double[] yhat;       // point prediction
        public final double[] yhatLower;  // lower bound
        public final double[] yhatUpper;  // upper bound

        public ForecastResult(double[] timestamps, double[] yhat,
                              double[] yhatLower, double[] yhatUpper) {
            this.timestamps = timestamps;
            this.yhat = yhat;
            this.yhatLower = yhatLower;
            this.yhatUpper = yhatUpper;
        }

        public int size() { return yhat.length; }
    }
}
