package com.weather.prophet;

import com.weather.prophet.data.DataPoint;
import com.weather.prophet.data.WeatherDataGenerator;
import com.weather.prophet.model.HolidayComponent;
import com.weather.prophet.model.HolidayComponent.Holiday;
import com.weather.prophet.model.ProphetModel;

import java.util.ArrayList;
import java.util.List;

/**
 * Weather Prophet - 天气预测模型
 *
 * A Java implementation inspired by Facebook's Prophet time series forecasting library.
 *
 * Core Model: y(t) = g(t) + s(t) + h(t) + ε(t)
 *
 * Components:
 *   g(t) - Piecewise linear trend with automatic changepoint detection
 *   s(t) - Fourier series seasonality (yearly, weekly cycles)
 *   h(t) - Holiday/special event effects
 *   ε(t) - Normally distributed error term
 *
 * This demo:
 *   1. Generates 365 days of synthetic temperature data
 *   2. Fits the Prophet model (trend + yearly/weekly seasonality + holidays)
 *   3. Forecasts the next 30 days with 80% confidence intervals
 *   4. Prints a detailed report with ASCII visualization
 */
public class WeatherProphetApp {

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("  Weather Prophet - 天气预测小模型");
        System.out.println("  Inspired by Facebook Prophet | Implemented in Java");
        System.out.println("=".repeat(70));
        System.out.println();

        // ============================================================
        // 1. Generate synthetic weather data
        // ============================================================
        System.out.println("[1] Generating synthetic weather data...");
        WeatherDataGenerator generator = new WeatherDataGenerator(42);

        int trainDays = 365;
        List<DataPoint> trainData = generator.generateTemperatureData(
                trainDays,
                15.0,   // base temperature: 15°C
                12.0,   // annual amplitude: ±12°C
                2.0,    // warming trend: 2°C per century (0.02/year)
                3.0     // noise level: 3°C std dev
        );

        System.out.printf("    Generated %d days of temperature data%n", trainData.size());
        System.out.printf("    Date range: Day 0 ~ Day %d%n", trainDays - 1);
        System.out.printf("    Temperature range: %.1f°C ~ %.1f°C%n",
                trainData.stream().mapToDouble(DataPoint::getValue).min().orElse(0),
                trainData.stream().mapToDouble(DataPoint::getValue).max().orElse(0));
        System.out.println();

        // ============================================================
        // 2. Define holidays / special events
        // ============================================================
        System.out.println("[2] Defining holiday effects...");
        List<Holiday> holidays = new ArrayList<>();
        // Simulate some weather-affecting events
        holidays.add(new Holiday("Spring Festival", 30, 3, 3));    // Late January
        holidays.add(new Holiday("Summer Heat Wave", 200, 2, 5));  // Mid July heat wave
        holidays.add(new Holiday("National Day", 273, 2, 2));      // October 1st
        holidays.add(new Holiday("Cold Snap", 15, 1, 3));          // Mid January cold
        HolidayComponent holidayComponent = new HolidayComponent(holidays);
        System.out.println("    Defined " + holidays.size() + " special weather events");
        System.out.println();

        // ============================================================
        // 3. Build and fit the Prophet model
        // ============================================================
        System.out.println("[3] Building and fitting Prophet model...");
        ProphetModel model = new ProphetModel(25);

        // Add seasonality components
        model.addSeasonality("yearly", 365.25, 10);  // 10 Fourier terms for annual cycle
        model.addSeasonality("weekly", 7, 3);         // 3 Fourier terms for weekly cycle

        // Set holidays
        model.setHolidays(holidayComponent);

        // Fit the model
        model.fit(trainData);
        System.out.println();

        // ============================================================
        // 4. Print model decomposition
        // ============================================================
        System.out.println("[4] Model decomposition:");
        System.out.println("-".repeat(50));
        System.out.println(model.getTrend());
        for (var season : model.getSeasonalities()) {
            System.out.println(season);
        }
        System.out.println(model.getHolidays());
        System.out.printf("  Residual std error: %.4f°C%n", model.getStdError());
        System.out.println();

        // ============================================================
        // 5. Forecast next 30 days
        // ============================================================
        System.out.println("[5] Forecasting next 30 days...");
        int forecastDays = 30;
        double[] futureTimestamps = new double[forecastDays];
        for (int i = 0; i < forecastDays; i++) {
            futureTimestamps[i] = trainDays + i;
        }

        ProphetModel.ForecastResult forecast = model.predict(futureTimestamps, 0.8);
        System.out.println();

        // ============================================================
        // 6. Print forecast results
        // ============================================================
        System.out.println("[6] Weather Forecast Results:");
        System.out.println("=".repeat(70));
        System.out.printf("  %-8s | %-10s | %-14s | %-10s%n",
                "Day", "Predicted", "80% CI", "Trend");
        System.out.println("-".repeat(70));

        double[] trendPred = model.getTrend().predict(futureTimestamps);
        for (int i = 0; i < forecastDays; i++) {
            System.out.printf("  Day %-4d | %7.2f°C  | [%6.2f, %6.2f] | %7.2f°C%n",
                    (int) futureTimestamps[i],
                    forecast.yhat[i],
                    forecast.yhatLower[i],
                    forecast.yhatUpper[i],
                    trendPred[i]);
        }
        System.out.println();

        // ============================================================
        // 7. ASCII Visualization
        // ============================================================
        System.out.println("[7] Temperature Forecast Visualization:");
        System.out.println("=".repeat(70));
        printASCIIChart(forecast, trainData);
        System.out.println();

        // ============================================================
        // 8. Model evaluation on training data
        // ============================================================
        System.out.println("[8] Model Evaluation (Training Data):");
        System.out.println("-".repeat(50));
        evaluateModel(model, trainData);
        System.out.println();

        // ============================================================
        // 9. Component contribution analysis
        // ============================================================
        System.out.println("[9] Component Contribution Analysis:");
        System.out.println("-".repeat(50));
        analyzeContributions(model, futureTimestamps);
        System.out.println();

        System.out.println("=".repeat(70));
        System.out.println("  Weather Prophet - Forecast Complete!");
        System.out.println("=".repeat(70));
    }

    /**
     * Print an ASCII chart of the forecast.
     */
    private static void printASCIIChart(ProphetModel.ForecastResult forecast,
                                         List<DataPoint> trainData) {
        // Find min/max for scaling
        double minVal = Double.MAX_VALUE, maxVal = -Double.MAX_VALUE;
        for (int i = 0; i < forecast.size(); i++) {
            minVal = Math.min(minVal, forecast.yhatLower[i]);
            maxVal = Math.max(maxVal, forecast.yhatUpper[i]);
        }
        // Also include last 14 days of training data for context
        int contextDays = 14;
        int trainStart = Math.max(0, trainData.size() - contextDays);
        for (int i = trainStart; i < trainData.size(); i++) {
            minVal = Math.min(minVal, trainData.get(i).getValue());
            maxVal = Math.max(maxVal, trainData.get(i).getValue());
        }

        int chartWidth = 50;
        int chartHeight = 20;
        double range = maxVal - minVal;
        if (range < 1) range = 1;

        // Build character grid
        char[][] grid = new char[chartHeight][chartWidth + contextDays + forecast.size()];
        for (int r = 0; r < chartHeight; r++) {
            for (int c = 0; c < grid[0].length; c++) {
                grid[r][c] = ' ';
            }
        }

        int totalCols = contextDays + forecast.size();

        // Plot training data (last N days)
        for (int i = trainStart; i < trainData.size(); i++) {
            int col = i - trainStart;
            int row = (int) ((maxVal - trainData.get(i).getValue()) / range * (chartHeight - 1));
            row = Math.max(0, Math.min(chartHeight - 1, row));
            grid[row][col] = '#';
        }

        // Plot forecast line
        for (int i = 0; i < forecast.size(); i++) {
            int col = contextDays + i;
            int row = (int) ((maxVal - forecast.yhat[i]) / range * (chartHeight - 1));
            row = Math.max(0, Math.min(chartHeight - 1, row));
            grid[row][col] = '*';
        }

        // Plot confidence interval bounds
        for (int i = 0; i < forecast.size(); i++) {
            int col = contextDays + i;
            int upperRow = (int) ((maxVal - forecast.yhatUpper[i]) / range * (chartHeight - 1));
            int lowerRow = (int) ((maxVal - forecast.yhatLower[i]) / range * (chartHeight - 1));
            upperRow = Math.max(0, Math.min(chartHeight - 1, upperRow));
            lowerRow = Math.max(0, Math.min(chartHeight - 1, lowerRow));
            if (grid[upperRow][col] == ' ') grid[upperRow][col] = '-';
            if (grid[lowerRow][col] == ' ') grid[lowerRow][col] = '-';
        }

        // Print chart with Y-axis labels
        System.out.printf("  %8s |", "Temp(°C)");
        for (int c = 0; c < contextDays; c++) System.out.print("-");
        for (int c = 0; c < forecast.size(); c++) System.out.print("+");
        System.out.println();

        for (int r = 0; r < chartHeight; r++) {
            double temp = maxVal - (double) r / (chartHeight - 1) * range;
            if (r == 0 || r == chartHeight / 2 || r == chartHeight - 1) {
                System.out.printf("  %7.1f°C |", temp);
            } else {
                System.out.printf("           |");
            }
            for (int c = 0; c < grid[0].length; c++) {
                System.out.print(grid[r][c]);
            }
            System.out.println();
        }

        // X-axis
        System.out.printf("           +");
        for (int c = 0; c < contextDays; c++) System.out.print("-");
        for (int c = 0; c < forecast.size(); c++) System.out.print("+");
        System.out.println();

        // Legend
        System.out.printf("           %-" + contextDays + "s%-30s%n",
                "<-- Training (" + contextDays + "d) -->",
                "<------- Forecast (" + forecast.size() + "d) ------->");
        System.out.println("  Legend: # = actual, * = predicted, - = 80% CI bounds");
    }

    /**
     * Evaluate model performance on training data.
     */
    private static void evaluateModel(ProphetModel model, List<DataPoint> data) {
        double[] timestamps = new double[data.size()];
        double[] actual = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            timestamps[i] = data.get(i).getTimestamp();
            actual[i] = data.get(i).getValue();
        }

        double[] predicted = model.predict(timestamps);

        double mae = 0, rmse = 0, mape = 0;
        double maxError = 0;
        int maxErrorIdx = 0;

        for (int i = 0; i < data.size(); i++) {
            double error = actual[i] - predicted[i];
            mae += Math.abs(error);
            rmse += error * error;
            if (Math.abs(actual[i]) > 0.1) {
                mape += Math.abs(error / actual[i]);
            }
            if (Math.abs(error) > maxError) {
                maxError = Math.abs(error);
                maxErrorIdx = i;
            }
        }

        int n = data.size();
        mae /= n;
        rmse = Math.sqrt(rmse / n);
        mape /= n;

        // R² score
        double meanActual = 0;
        for (double v : actual) meanActual += v;
        meanActual /= n;

        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++) {
            ssTot += (actual[i] - meanActual) * (actual[i] - meanActual);
            ssRes += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
        }
        double r2 = 1 - ssRes / ssTot;

        System.out.printf("  Mean Absolute Error (MAE):     %.4f°C%n", mae);
        System.out.printf("  Root Mean Square Error (RMSE): %.4f°C%n", rmse);
        System.out.printf("  Mean Abs Percentage Error:     %.2f%%%n", mape * 100);
        System.out.printf("  R² Score:                      %.6f%n", r2);
        System.out.printf("  Max Error:                     %.4f°C (at Day %d)%n", maxError, maxErrorIdx);
    }

    /**
     * Analyze the contribution of each model component.
     */
    private static void analyzeContributions(ProphetModel model, double[] timestamps) {
        double[] trendPred = model.getTrend().predict(timestamps);
        double[] totalPred = model.predict(timestamps);

        // Compute average contribution of each component
        double avgTrend = 0, avgSeasonal = 0, avgHoliday = 0;
        int n = timestamps.length;

        for (int i = 0; i < n; i++) {
            avgTrend += trendPred[i];
            double seasonalContrib = 0;
            for (var season : model.getSeasonalities()) {
                double[] sp = season.predict(timestamps);
                seasonalContrib += sp[i];
            }
            avgSeasonal += seasonalContrib;
        }

        avgTrend /= n;
        avgSeasonal /= n;
        avgHoliday = 0;
        if (model.getHolidays() != null) {
            double[] hp = model.getHolidays().predict(timestamps);
            for (double v : hp) avgHoliday += v;
            avgHoliday /= n;
        }

        double totalAvg = avgTrend + avgSeasonal + avgHoliday;

        System.out.printf("  %-20s | %-10s | %-10s%n", "Component", "Avg Value", "Contribution");
        System.out.println("  " + "-".repeat(45));
        System.out.printf("  %-20s | %8.2f°C | %8.1f%%%n",
                "Trend", avgTrend, avgTrend / totalAvg * 100);
        System.out.printf("  %-20s | %8.2f°C | %8.1f%%%n",
                "Seasonality (total)", avgSeasonal,
                Math.abs(avgSeasonal) / Math.abs(totalAvg) * 100);

        for (var season : model.getSeasonalities()) {
            double[] sp = season.predict(timestamps);
            double avg = 0;
            for (double v : sp) avg += v;
            avg /= n;
            System.out.printf("    %-18s | %8.2f°C |%n", season.getName(), avg);
        }

        System.out.printf("  %-20s | %8.2f°C | %8.1f%%%n",
                "Holidays", avgHoliday,
                Math.abs(avgHoliday) / Math.abs(totalAvg) * 100);
        System.out.println("  " + "-".repeat(45));
        System.out.printf("  %-20s | %8.2f°C | %8.1f%%%n",
                "Total", totalAvg, 100.0);
    }
}
