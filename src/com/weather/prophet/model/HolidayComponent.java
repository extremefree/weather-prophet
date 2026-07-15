package com.weather.prophet.model;

import com.weather.prophet.data.DataPoint;
import com.weather.prophet.matrix.Matrix;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Holiday / special event component, inspired by Prophet's holiday modeling.
 *
 * Models the effect of specific dates (holidays, events) on the time series.
 * Each holiday can have a window of influence (days before and after).
 *
 * h(t) = Σ_{j} κ_j * 1(t ∈ holiday_j_window)
 *
 * Where κ_j is the effect size of holiday j, estimated from data.
 */
public class HolidayComponent {

    /**
     * Represents a single holiday or special event.
     */
    public static class Holiday {
        public final String name;
        public final double timestamp;  // day number
        public final int windowBefore;  // days of influence before
        public final int windowAfter;   // days of influence after

        public Holiday(String name, double timestamp, int windowBefore, int windowAfter) {
            this.name = name;
            this.timestamp = timestamp;
            this.windowBefore = windowBefore;
            this.windowAfter = windowAfter;
        }

        public Holiday(String name, double timestamp) {
            this(name, timestamp, 1, 1);
        }
    }

    private final List<Holiday> holidays;
    private double[] effects; // effect size for each holiday

    public HolidayComponent(List<Holiday> holidays) {
        this.holidays = holidays;
    }

    /**
     * Build the holiday feature matrix.
     * Each column represents one holiday's window effect.
     */
    public Matrix buildHolidayFeatures(double[] timestamps) {
        int n = timestamps.length;
        int numHolidays = holidays.size();
        Matrix X = new Matrix(n, numHolidays);

        for (int i = 0; i < n; i++) {
            double t = timestamps[i];
            for (int j = 0; j < numHolidays; j++) {
                Holiday h = holidays.get(j);
                double diff = t - h.timestamp;
                if (diff >= -h.windowBefore && diff <= h.windowAfter) {
                    // Gaussian-decayed effect within window
                    double sigma = (h.windowBefore + h.windowAfter) / 2.0;
                    double effect = Math.exp(-0.5 * (diff * diff) / (sigma * sigma));
                    X.set(i, j, effect);
                }
            }
        }
        return X;
    }

    /**
     * Fit holiday effects from residuals.
     *
     * @param data        original data points
     * @param trendValues trend values to subtract
     * @param seasonalityValues seasonality values to subtract (can be null)
     */
    public void fit(List<DataPoint> data, double[] trendValues, double[] seasonalityValues) {
        int n = data.size();
        double[] timestamps = new double[n];
        double[] residuals = new double[n];

        for (int i = 0; i < n; i++) {
            timestamps[i] = data.get(i).getTimestamp();
            double residual = data.get(i).getValue() - trendValues[i];
            if (seasonalityValues != null) {
                residual -= seasonalityValues[i];
            }
            residuals[i] = residual;
        }

        Matrix X = buildHolidayFeatures(timestamps);
        effects = Matrix.solveLeastSquares(X, residuals, 1.0);
    }

    /**
     * Predict holiday effects at given timestamps.
     */
    public double[] predict(double[] timestamps) {
        if (effects == null) {
            throw new IllegalStateException("Model not fitted yet");
        }

        Matrix X = buildHolidayFeatures(timestamps);
        double[] result = new double[timestamps.length];

        for (int i = 0; i < timestamps.length; i++) {
            double value = 0;
            for (int j = 0; j < effects.length; j++) {
                value += X.get(i, j) * effects[j];
            }
            result[i] = value;
        }
        return result;
    }

    /**
     * Get a map of holiday name to its estimated effect.
     */
    public Map<String, Double> getHolidayEffects() {
        Map<String, Double> map = new HashMap<>();
        if (effects != null) {
            for (int i = 0; i < holidays.size(); i++) {
                map.put(holidays.get(i).name, effects[i]);
            }
        }
        return map;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("HolidayComponent(holidays=%d)\n", holidays.size()));
        if (effects != null) {
            Map<String, Double> map = getHolidayEffects();
            for (Map.Entry<String, Double> entry : map.entrySet()) {
                sb.append(String.format("  %s: %.4f°C\n", entry.getKey(), entry.getValue()));
            }
        }
        return sb.toString();
    }
}
