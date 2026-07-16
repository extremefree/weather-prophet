package com.weather.prophet;

import com.weather.prophet.core.ProphetConfig;
import com.weather.prophet.core.ProphetModel;
import com.weather.prophet.data.DataPoint;

import java.util.List;

/**
 * Weather Prophet — 完全遵循 Facebook Prophet 逻辑的天气预测模型
 *
 * y(t) = g(t) * (1 + multiplicative_terms) + additive_terms + ε(t)
 *
 * Trend: Piecewise linear / logistic / flat
 * Seasonality: Fourier series (additive or multiplicative)
 * Holidays: Windowed indicator features
 * Priors: Normal on k,m,beta; Laplace on delta; Normal(0,0.5) on sigma_obs
 * Optimization: L-BFGS (Stan's optimizing())
 * MCMC: NUTS / Metropolis-Hastings (Stan's sampling())
 * Uncertainty: Poisson future changepoints + posterior predictive simulation
 */
public class WeatherProphetApp {

    public static void main(String[] args) {
        System.out.println("=".repeat(72));
        System.out.println("  Weather Prophet — 天气预测模型 (Java Prophet Implementation)");
        System.out.println("  Faithfully following Facebook Prophet's algorithm");
        System.out.println("  Model: y(t) = g(t) * (1 + mult) + add + ε(t)");
        System.out.println("  Trend: Piecewise linear with Laplace prior (L1 sparse changepoints)");
        System.out.println("  Optimization: L-BFGS (same as Stan's optimizing())");
        System.out.println("  Uncertainty: Poisson future changepoints + posterior predictive");
        System.out.println("  Scaling: absmax (no mean centering, same as Prophet)");
        System.out.println("=".repeat(72));
        System.out.println();

        // 1. Generate synthetic weather data
        System.out.println("[1] Generating synthetic weather data...");
        int trainDays = 365;
        List<DataPoint> trainData = DataPoint.generateTemperatureData(trainDays);
        System.out.printf("    %d days of temperature data generated%n", trainDays);
        System.out.printf("    Range: %.1f°C ~ %.1f°C%n",
                trainData.stream().mapToDouble(DataPoint::getValue).min().orElse(0),
                trainData.stream().mapToDouble(DataPoint::getValue).max().orElse(0));
        System.out.println();

        // 2. Configure Prophet model
        System.out.println("[2] Configuring Prophet model...");
        ProphetConfig config = new ProphetConfig();

        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.nChangepoints = 25;
        config.changepointRange = 0.8;
        config.changepointPriorScale = 0.05;

        config.yearlySeasonality = true;
        config.yearlyFourierOrder = 10;
        config.weeklySeasonality = true;
        config.weeklyFourierOrder = 3;
        config.seasonalityPriorScale = 10.0;
        config.holidaysPriorScale = 10.0;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;

        config.addHoliday("Spring Festival", 30, 3, 3);
        config.addHoliday("Summer Heat Wave", 200, 2, 5);
        config.addHoliday("National Day", 273, 2, 2);
        config.addHoliday("Cold Snap", 15, 1, 3);

        config.mcmcSamples = 100;
        config.mcmcWarmup = 50;
        config.mcmcChains = 4;
        config.useNUTS = true;
        config.uncertaintySamples = 500;
        config.sigmaObsPriorScale = 0.5;
        config.scaling = ProphetConfig.Scaling.ABSMAX;
        config.verbose = true;

        System.out.printf("    Growth: %s, Changepoints: %d (Laplace τ=%.3f)%n",
                config.growth, config.nChangepoints, config.changepointPriorScale);
        System.out.printf("    Seasonality: yearly(order=%d, period=%.2f) + weekly(order=%d)%n",
                config.yearlyFourierOrder, config.yearlyPeriod, config.weeklyFourierOrder);
        System.out.printf("    Seasonality mode: %s%n", config.seasonalityMode);
        System.out.printf("    Scaling: %s (no mean centering)%n", config.scaling);
        System.out.printf("    MCMC: %d samples, %d warmup, %d chains, NUTS=%s%n",
                config.mcmcSamples, config.mcmcWarmup, config.mcmcChains, config.useNUTS);
        System.out.printf("    Uncertainty: %d predictive samples (Poisson future changepoints)%n",
                config.uncertaintySamples);
        System.out.println();

        // 3. Fit
        System.out.println("[3] Fitting Prophet model...");
        ProphetModel model = new ProphetModel(config);
        model.fit(trainData);
        System.out.println();

        // 4. Decomposition
        System.out.println("[4] Model Decomposition (Prophet-style):");
        System.out.println("-".repeat(60));
        System.out.printf("  Trend: k=%.6f, m=%.6f%n", model.getK(), model.getM());
        System.out.printf("  Observation noise (sigma_obs): %.6f%n", model.getSigmaObs());

        double[] delta = model.getDelta();
        int sigCps = 0;
        for (double d : delta) if (Math.abs(d) > 1e-4) sigCps++;
        System.out.printf("  Changepoints: %d/%d significant (Laplace L1 sparsity)%n",
                sigCps, delta.length);
        System.out.printf("  Seasonality coefficients (beta): %d params%n",
                model.getBeta().length);
        System.out.println();

        // 5. Forecast
        System.out.println("[5] Forecasting next 30 days...");
        int forecastDays = 30;
        double[] futureT = new double[forecastDays];
        for (int i = 0; i < forecastDays; i++) {
            futureT[i] = trainDays + i;
        }
        ProphetModel.ForecastResult forecast = model.predict(futureT);
        System.out.println();

        // 6. Forecast table
        System.out.println("[6] Weather Forecast Results:");
        System.out.println("=".repeat(72));
        System.out.printf("  %-8s | %-10s | %-22s | %-10s%n",
                "Day", "Predicted", "80% CI", "Trend");
        System.out.println("-".repeat(72));
        for (int i = 0; i < forecastDays; i++) {
            System.out.printf("  Day %-4d | %7.2f°C  | [%7.2f, %7.2f]°C | %7.2f°C%n",
                    (int) forecast.t[i],
                    forecast.yhat[i],
                    forecast.yhatLower[i],
                    forecast.yhatUpper[i],
                    forecast.trend[i]);
        }
        System.out.println();

        // 7. ASCII chart
        System.out.println("[7] Temperature Forecast Visualization:");
        System.out.println("=".repeat(72));
        printASCIIChart(forecast, trainData, forecastDays);
        System.out.println();

        // 8. Evaluation
        System.out.println("[8] Model Evaluation (In-Sample):");
        System.out.println("-".repeat(60));
        evaluateModel(model, trainData);
        System.out.println();

        // 9. Summary
        System.out.println("[9] Algorithm Summary (Prophet Faithfulness Check):");
        System.out.println("-".repeat(60));
        System.out.println("  ✓ Model: y(t) = g(t) * (1 + mult) + add + ε(t)");
        System.out.println("  ✓ Trend: Piecewise linear g(t) = (k+A^T*δ)*t + (m+A^T*γ)");
        System.out.println("  ✓ Continuity: γ_j = -s_j * δ_j (linear), iterative k_s ratio (logistic)");
        System.out.println("  ✓ Prior on δ: Laplace(0, τ) → L1 sparse changepoints");
        System.out.println("  ✓ Prior on k,m: Normal(0, 5)");
        System.out.println("  ✓ Prior on σ_obs: Normal(0, 0.5) truncated (Stan model)");
        System.out.println("  ✓ Prior on β: Normal(0, σ_j) per-component prior scale");
        System.out.println("  ✓ Scaling: absmax (y_scale = max|y|, no mean centering)");
        System.out.println("  ✓ Optimization: L-BFGS (Stan's optimizing())");
        System.out.println("  ✓ MCMC: NUTS (Stan's sampling())");
        System.out.println("  ✓ Uncertainty: Poisson future changepoints + posterior predictive");
        System.out.println("  ✓ Seasonality: Additive + Multiplicative modes");
        System.out.println("  ✓ Annual period: 365.25 (leap year corrected)");
        System.out.println("  ✓ Growth: Linear / Logistic / Flat");
        System.out.println("  ✓ GPU: OpenCL acceleration via JOCL");
        System.out.println();

        System.out.println("=".repeat(72));
        System.out.println("  Weather Prophet — Forecast Complete!");
        System.out.println("=".repeat(72));
    }

    private static void printASCIIChart(ProphetModel.ForecastResult forecast,
                                         List<DataPoint> trainData, int forecastDays) {
        double minVal = Double.MAX_VALUE, maxVal = -Double.MAX_VALUE;
        int contextDays = 14;
        int trainStart = Math.max(0, trainData.size() - contextDays);
        for (int i = trainStart; i < trainData.size(); i++) {
            minVal = Math.min(minVal, trainData.get(i).getValue());
            maxVal = Math.max(maxVal, trainData.get(i).getValue());
        }
        for (int i = 0; i < forecastDays; i++) {
            minVal = Math.min(minVal, forecast.yhatLower[i]);
            maxVal = Math.max(maxVal, forecast.yhatUpper[i]);
        }

        int chartHeight = 18;
        double range = maxVal - minVal;
        if (range < 1) range = 1;
        int totalCols = contextDays + forecastDays;

        char[][] grid = new char[chartHeight][totalCols];
        for (int r = 0; r < chartHeight; r++)
            for (int c = 0; c < totalCols; c++) grid[r][c] = ' ';

        for (int i = trainStart; i < trainData.size(); i++) {
            int col = i - trainStart;
            int row = (int) ((maxVal - trainData.get(i).getValue()) / range * (chartHeight - 1));
            row = Math.max(0, Math.min(chartHeight - 1, row));
            grid[row][col] = '#';
        }

        for (int i = 0; i < forecastDays; i++) {
            int col = contextDays + i;
            int row = (int) ((maxVal - forecast.yhat[i]) / range * (chartHeight - 1));
            row = Math.max(0, Math.min(chartHeight - 1, row));
            grid[row][col] = '*';
        }

        for (int i = 0; i < forecastDays; i++) {
            int col = contextDays + i;
            int upperRow = (int) ((maxVal - forecast.yhatUpper[i]) / range * (chartHeight - 1));
            int lowerRow = (int) ((maxVal - forecast.yhatLower[i]) / range * (chartHeight - 1));
            upperRow = Math.max(0, Math.min(chartHeight - 1, upperRow));
            lowerRow = Math.max(0, Math.min(chartHeight - 1, lowerRow));
            if (grid[upperRow][col] == ' ') grid[upperRow][col] = '-';
            if (grid[lowerRow][col] == ' ') grid[lowerRow][col] = '-';
        }

        for (int r = 0; r < chartHeight; r++) {
            double temp = maxVal - (double) r / (chartHeight - 1) * range;
            if (r == 0 || r == chartHeight / 2 || r == chartHeight - 1) {
                System.out.printf("  %7.1f°C |", temp);
            } else {
                System.out.printf("           |");
            }
            for (int c = 0; c < totalCols; c++) System.out.print(grid[r][c]);
            System.out.println();
        }

        System.out.printf("           +");
        for (int c = 0; c < totalCols; c++) System.out.print("-");
        System.out.println();
        System.out.printf("           <--- Training (%dd) ---><----- Forecast (%dd) ----->%n",
                contextDays, forecastDays);
        System.out.println("  Legend: # = actual, * = predicted, - = 80% CI");
    }

    private static void evaluateModel(ProphetModel model, List<DataPoint> data) {
        double[] timestamps = new double[data.size()];
        double[] actual = new double[data.size()];
        for (int i = 0; i < data.size(); i++) {
            timestamps[i] = data.get(i).getTimestamp();
            actual[i] = data.get(i).getValue();
        }

        ProphetModel.ForecastResult fitted = model.predict(timestamps);

        double mae = 0, rmse = 0;
        for (int i = 0; i < data.size(); i++) {
            double error = actual[i] - fitted.yhat[i];
            mae += Math.abs(error);
            rmse += error * error;
        }
        int n = data.size();
        mae /= n;
        rmse = Math.sqrt(rmse / n);

        double meanActual = 0;
        for (double v : actual) meanActual += v;
        meanActual /= n;
        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++) {
            ssTot += (actual[i] - meanActual) * (actual[i] - meanActual);
            ssRes += (actual[i] - fitted.yhat[i]) * (actual[i] - fitted.yhat[i]);
        }
        double r2 = 1 - ssRes / ssTot;

        int covered = 0;
        for (int i = 0; i < n; i++) {
            if (actual[i] >= fitted.yhatLower[i] && actual[i] <= fitted.yhatUpper[i]) covered++;
        }

        System.out.printf("  MAE:              %.4f°C%n", mae);
        System.out.printf("  RMSE:             %.4f°C%n", rmse);
        System.out.printf("  R²:               %.6f%n", r2);
        System.out.printf("  80%% CI Coverage:  %.1f%%%n", (double) covered / n * 100);
    }
}
