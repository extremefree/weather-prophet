package com.weather.prophet.model;

import com.weather.prophet.data.DataPoint;
import com.weather.prophet.matrix.Matrix;

import java.util.List;

/**
 * Seasonality component using Fourier series, inspired by Prophet's seasonality modeling.
 *
 * The seasonality is modeled as:
 *   s(t) = Σ_{n=1}^{N} [a_n * cos(2π*n*t/P) + b_n * sin(2π*n*t/P)]
 *
 * Where:
 *   - P is the period (e.g., 365.25 for yearly, 7 for weekly)
 *   - N is the Fourier order (number of harmonics)
 *   - a_n, b_n are coefficients fitted via least squares
 *
 * Supports multiple seasonality types: yearly, weekly, daily.
 */
public class SeasonalityComponent {

    private final String name;
    private final double period;    // in days
    private final int fourierOrder; // number of harmonics
    private double[] coefficients;  // [a_1, b_1, a_2, b_2, ..., a_N, b_N]

    /**
     * @param name         descriptive name (e.g., "yearly", "weekly")
     * @param period       period in days (365.25 for yearly, 7 for weekly)
     * @param fourierOrder number of Fourier terms (higher = more flexible)
     */
    public SeasonalityComponent(String name, double period, int fourierOrder) {
        this.name = name;
        this.period = period;
        this.fourierOrder = fourierOrder;
    }

    /**
     * Build the Fourier feature matrix for given timestamps.
     * Each pair of columns represents one harmonic: [cos(2π*n*t/P), sin(2π*n*t/P)]
     *
     * @param timestamps array of timestamps
     * @return feature matrix of shape (len(timestamps), 2 * fourierOrder)
     */
    public Matrix buildFourierFeatures(double[] timestamps) {
        int n = timestamps.length;
        int numFeatures = 2 * fourierOrder;
        Matrix X = new Matrix(n, numFeatures);

        for (int i = 0; i < n; i++) {
            double t = timestamps[i];
            for (int k = 1; k <= fourierOrder; k++) {
                double angle = 2.0 * Math.PI * k * t / period;
                X.set(i, 2 * (k - 1), Math.cos(angle));
                X.set(i, 2 * (k - 1) + 1, Math.sin(angle));
            }
        }
        return X;
    }

    /**
     * Fit the seasonality component to the residuals (data after removing trend).
     *
     * @param data      original data points
     * @param trendValues trend component values to subtract
     */
    public void fit(List<DataPoint> data, double[] trendValues) {
        int n = data.size();
        double[] timestamps = new double[n];
        double[] residuals = new double[n];

        for (int i = 0; i < n; i++) {
            timestamps[i] = data.get(i).getTimestamp();
            residuals[i] = data.get(i).getValue() - trendValues[i];
        }

        Matrix X = buildFourierFeatures(timestamps);
        coefficients = Matrix.solveLeastSquares(X, residuals, 0.1);
    }

    /**
     * Fit seasonality from raw data (assumes trend is zero).
     */
    public void fit(List<DataPoint> data) {
        double[] zeros = new double[data.size()];
        fit(data, zeros);
    }

    /**
     * Predict the seasonality component at given timestamps.
     *
     * @param timestamps array of timestamps
     * @return array of seasonality values
     */
    public double[] predict(double[] timestamps) {
        if (coefficients == null) {
            throw new IllegalStateException("Model not fitted yet");
        }

        Matrix X = buildFourierFeatures(timestamps);
        double[] result = new double[timestamps.length];

        for (int i = 0; i < timestamps.length; i++) {
            double value = 0;
            for (int j = 0; j < coefficients.length; j++) {
                value += X.get(i, j) * coefficients[j];
            }
            result[i] = value;
        }
        return result;
    }

    /**
     * Get the amplitude of each harmonic.
     * amplitude_n = sqrt(a_n^2 + b_n^2)
     */
    public double[] getHarmonicAmplitudes() {
        if (coefficients == null) return new double[0];
        double[] amplitudes = new double[fourierOrder];
        for (int k = 0; k < fourierOrder; k++) {
            double a = coefficients[2 * k];
            double b = coefficients[2 * k + 1];
            amplitudes[k] = Math.sqrt(a * a + b * b);
        }
        return amplitudes;
    }

    public String getName() { return name; }
    public double getPeriod() { return period; }
    public int getFourierOrder() { return fourierOrder; }
    public double[] getCoefficients() { return coefficients; }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("SeasonalityComponent(name=%s, period=%.2f, order=%d)\n",
                name, period, fourierOrder));
        if (coefficients != null) {
            double[] amps = getHarmonicAmplitudes();
            sb.append("  Harmonic amplitudes: [");
            for (int i = 0; i < Math.min(amps.length, 5); i++) {
                sb.append(String.format("%.4f", amps[i]));
                if (i < amps.length - 1 && i < 4) sb.append(", ");
            }
            if (amps.length > 5) sb.append(", ...");
            sb.append("]\n");
        }
        return sb.toString();
    }
}
