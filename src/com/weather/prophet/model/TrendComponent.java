package com.weather.prophet.model;

import com.weather.prophet.data.DataPoint;
import com.weather.prophet.matrix.Matrix;

import java.util.ArrayList;
import java.util.List;

/**
 * Piecewise linear trend component, inspired by Prophet's trend modeling.
 *
 * The trend is modeled as:
 *   g(t) = (k + A(t)^T * delta) * t + (m + A(t)^T * gamma)
 *
 * Where:
 *   - k is the base growth rate
 *   - m is the base offset
 *   - delta is the vector of rate changes at each changepoint
 *   - gamma is the offset adjustment for continuity
 *   - A(t) is a matrix indicating which changepoints are active at time t
 *
 * This implementation automatically places changepoints at uniform intervals
 * and fits the rates via least-squares optimization.
 */
public class TrendComponent {

    private final int numChangepoints;
    private double[] changepoints; // timestamps of changepoints
    private double baseRate;       // k
    private double baseOffset;     // m
    private double[] deltas;       // rate changes at each changepoint

    /**
     * @param numChangepoints number of changepoints to place (default 25)
     */
    public TrendComponent(int numChangepoints) {
        this.numChangepoints = numChangepoints;
    }

    public TrendComponent() {
        this(25);
    }

    /**
     * Fit the trend component to the data.
     *
     * @param data list of DataPoints
     */
    public void fit(List<DataPoint> data) {
        int n = data.size();
        if (n < 2) throw new IllegalArgumentException("Need at least 2 data points");

        double tMin = data.get(0).getTimestamp();
        double tMax = data.get(n - 1).getTimestamp();
        double T = tMax - tMin;

        // Place changepoints uniformly in the first 80% of the time range
        // (same as Prophet's default)
        changepoints = new double[numChangepoints];
        double step = (0.8 * T) / numChangepoints;
        for (int i = 0; i < numChangepoints; i++) {
            changepoints[i] = tMin + (i + 1) * step;
        }

        // Build the design matrix for the trend
        // Features: [1, t, A(t)_1, A(t)_2, ..., A(t)_S]
        // where A(t)_j = max(0, t - s_j) for changepoint s_j
        int numFeatures = 2 + numChangepoints; // intercept, slope, changepoint adjustments
        Matrix X = new Matrix(n, numFeatures);

        for (int i = 0; i < n; i++) {
            double t = data.get(i).getTimestamp();
            X.set(i, 0, 1.0); // intercept
            X.set(i, 1, t);   // linear trend
            for (int j = 0; j < numChangepoints; j++) {
                X.set(i, 2 + j, Math.max(0, t - changepoints[j]));
            }
        }

        // Target values
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = data.get(i).getValue();
        }

        // Solve via least squares with ridge regularization
        double[] beta = Matrix.solveLeastSquares(X, y, 1.0);

        baseOffset = beta[0];
        baseRate = beta[1];
        deltas = new double[numChangepoints];
        for (int j = 0; j < numChangepoints; j++) {
            deltas[j] = beta[2 + j];
        }
    }

    /**
     * Predict the trend component at given timestamps.
     *
     * @param timestamps array of timestamps to predict
     * @return array of trend values
     */
    public double[] predict(double[] timestamps) {
        if (changepoints == null) {
            throw new IllegalStateException("Model not fitted yet");
        }

        double[] result = new double[timestamps.length];
        for (int i = 0; i < timestamps.length; i++) {
            double t = timestamps[i];
            double trend = baseRate * t + baseOffset;
            for (int j = 0; j < numChangepoints; j++) {
                trend += deltas[j] * Math.max(0, t - changepoints[j]);
            }
            result[i] = trend;
        }
        return result;
    }

    /**
     * Get the effective growth rate at time t.
     */
    public double getRateAt(double t) {
        double rate = baseRate;
        for (int j = 0; j < numChangepoints; j++) {
            if (t >= changepoints[j]) {
                rate += deltas[j];
            }
        }
        return rate;
    }

    public double getBaseRate() { return baseRate; }
    public double getBaseOffset() { return baseOffset; }
    public double[] getDeltas() { return deltas; }
    public double[] getChangepoints() { return changepoints; }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("TrendComponent(baseRate=%.6f, baseOffset=%.4f)\n", baseRate, baseOffset));
        sb.append(String.format("  Changepoints: %d\n", numChangepoints));
        if (deltas != null) {
            int significantCount = 0;
            for (double d : deltas) {
                if (Math.abs(d) > 0.001) significantCount++;
            }
            sb.append(String.format("  Significant rate changes: %d\n", significantCount));
        }
        return sb.toString();
    }
}
