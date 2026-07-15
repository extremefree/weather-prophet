package com.weather.prophet.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Data point and synthetic weather data generator.
 */
public class DataPoint {
    private final double timestamp;
    private final double value;

    public DataPoint(double timestamp, double value) {
        this.timestamp = timestamp;
        this.value = value;
    }

    public double getTimestamp() { return timestamp; }
    public double getValue() { return value; }

    /**
     * Generate synthetic daily temperature data with realistic patterns:
     * - Annual sinusoidal cycle (summer hot, winter cold)
     * - Weekly variation (prophet's weekly seasonality)
     * - Linear warming trend
     * - Gaussian noise
     */
    public static List<DataPoint> generateTemperatureData(int days, double baseTemp,
            double amplitude, double trendRate, double noiseLevel, long seed) {
        Random rng = new Random(seed);
        List<DataPoint> data = new ArrayList<>();
        double trendPerDay = trendRate / 365.0;

        for (int d = 0; d < days; d++) {
            double t = d;
            double dayOfYear = d % 365;

            double annualCycle = amplitude * Math.cos(2 * Math.PI * (dayOfYear - 182) / 365.0);
            double weeklyEffect = 1.5 * Math.sin(2 * Math.PI * (d % 7) / 7.0);
            double trend = baseTemp + trendPerDay * t;
            double noise = rng.nextGaussian() * noiseLevel;

            data.add(new DataPoint(t, trend + annualCycle + weeklyEffect + noise));
        }
        return data;
    }

    public static List<DataPoint> generateTemperatureData(int days) {
        return generateTemperatureData(days, 15.0, 12.0, 0.02, 3.0, 42L);
    }
}
