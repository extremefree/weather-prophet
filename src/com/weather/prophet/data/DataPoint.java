package com.weather.prophet.data;

/**
 * Represents a single data point with timestamp and observed value.
 */
public class DataPoint {
    private final double timestamp; // days from epoch
    private final double value;

    public DataPoint(double timestamp, double value) {
        this.timestamp = timestamp;
        this.value = value;
    }

    public double getTimestamp() {
        return timestamp;
    }

    public double getValue() {
        return value;
    }

    @Override
    public String toString() {
        return String.format("DataPoint(t=%.2f, y=%.2f)", timestamp, value);
    }
}
