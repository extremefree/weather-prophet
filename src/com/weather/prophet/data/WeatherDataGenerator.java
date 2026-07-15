package com.weather.prophet.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates synthetic weather data with realistic patterns:
 * - Annual temperature cycle (sinusoidal)
 * - Weekly variation
 * - Long-term warming trend
 * - Random noise
 *
 * This simulates real weather data for testing the Prophet model.
 */
public class WeatherDataGenerator {

    private final Random random;

    public WeatherDataGenerator(long seed) {
        this.random = new Random(seed);
    }

    /**
     * Generate synthetic daily temperature data.
     *
     * @param days number of days of historical data
     * @param baseTemp base temperature (Celsius)
     * @param amplitude annual temperature amplitude
     * @param trendRate warming trend per year (Celsius/year)
     * @param noiseLevel standard deviation of random noise
     * @return list of DataPoints
     */
    public List<DataPoint> generateTemperatureData(
            int days, double baseTemp, double amplitude,
            double trendRate, double noiseLevel) {

        List<DataPoint> data = new ArrayList<>();
        double trendPerDay = trendRate / 365.0;

        for (int d = 0; d < days; d++) {
            double t = d;
            double dayOfYear = d % 365;

            // Annual seasonality: peaks in summer (~day 182), troughs in winter
            double annualCycle = amplitude * Math.cos(2 * Math.PI * (dayOfYear - 182) / 365.0);

            // Weekly pattern: slight weekend effect (people perceive temps differently)
            double dayOfWeek = d % 7;
            double weeklyEffect = 1.5 * Math.sin(2 * Math.PI * dayOfWeek / 7.0);

            // Linear trend (gradual warming)
            double trend = baseTemp + trendPerDay * t;

            // Random noise (weather variability)
            double noise = random.nextGaussian() * noiseLevel;

            double value = trend + annualCycle + weeklyEffect + noise;
            data.add(new DataPoint(t, value));
        }
        return data;
    }

    /**
     * Generate synthetic humidity data (inversely correlated with temperature).
     */
    public List<DataPoint> generateHumidityData(int days, double baseHumidity) {
        List<DataPoint> data = new ArrayList<>();
        for (int d = 0; d < days; d++) {
            double dayOfYear = d % 365;
            // Humidity inversely correlated with temperature
            double seasonalHumidity = -15 * Math.cos(2 * Math.PI * (dayOfYear - 182) / 365.0);
            double noise = random.nextGaussian() * 5;
            data.add(new DataPoint(d, baseHumidity + seasonalHumidity + noise));
        }
        return data;
    }
}
