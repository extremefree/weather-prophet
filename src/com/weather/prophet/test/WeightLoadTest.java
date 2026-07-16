package com.weather.prophet.test;

import com.weather.prophet.core.ModelSerializer;
import com.weather.prophet.core.ProphetConfig;
import com.weather.prophet.core.ProphetModel;
import com.weather.prophet.data.DataPoint;

import java.util.*;

/**
 * Weight Loading Validation Test
 *
 * Strategy: For each model, generate the SAME training data used during training,
 * then compare two predictions:
 *   A) fit() → predict()    (freshly trained)
 *   B) load() → predict()   (loaded from saved weights)
 *
 * If serialization is correct, A and B should produce identical results.
 * We also test on FUTURE data (beyond training window) to validate
 * that loaded models generalize correctly.
 */
public class WeightLoadTest {

    static final long SEED = 42;

    public static void main(String[] args) throws Exception {
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║   WeatherProphet Weight Loading Consistency & Accuracy Test    ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝\n");

        int pass = 0, fail = 0;
        StringBuilder summary = new StringBuilder();

        // ── Test 1: Beijing ──
        {
            String name = "北京四季温度";
            String file = "beijing_temperature.model";
            int trainDays = 1826;
            int testDays = 60;
            double noiseStd = 3.0;

            // Generate training data (same as TrainModels)
            List<DataPoint> trainData = genBeijing(trainDays);
            // Generate future data
            List<DataPoint> futureData = genBeijing(testDays + trainDays);
            List<DataPoint> testData = futureData.subList(trainDays, futureData.size());

            double[] futureT = new double[testDays];
            double[] actualY = new double[testDays];
            for (int i = 0; i < testDays; i++) {
                futureT[i] = trainDays + i;
                actualY[i] = testData.get(i).getValue();
            }

            // A) Fresh train + predict
            ProphetConfig config = makeBeijingConfig();
            ProphetModel fresh = new ProphetModel(config);
            fresh.fit(trainData);
            ProphetModel.ForecastResult freshResult = fresh.predict(futureT);

            // B) Load weights + predict
            ProphetModel loaded = ModelSerializer.load("models/" + file);
            ProphetModel.ForecastResult loadResult = loaded.predict(futureT);

            // Compare consistency (A vs B)
            double maxDiff = 0;
            for (int i = 0; i < testDays; i++) {
                double diff = Math.abs(freshResult.yhat[i] - loadResult.yhat[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            // Evaluate accuracy (loaded model vs actual)
            double mae = 0, rmse = 0;
            int ciCount = 0;
            for (int i = 0; i < testDays; i++) {
                double err = Math.abs(loadResult.yhat[i] - actualY[i]);
                mae += err;
                rmse += err * err;
                if (actualY[i] >= loadResult.yhatLower[i] && actualY[i] <= loadResult.yhatUpper[i])
                    ciCount++;
            }
            mae /= testDays;
            rmse = Math.sqrt(rmse / testDays);
            double ci80 = ciCount * 100.0 / testDays;
            double maeRatio = mae / noiseStd;
            boolean passConsistency = maxDiff < 1.0;
            boolean passAccuracy = maeRatio < 1.5;
            boolean ok = passConsistency && passAccuracy;

            printResult(name, ok, maxDiff, mae, rmse, maeRatio, ci80, passConsistency, passAccuracy);
            summary.append(String.format("║ %-22s  %s  maxDiff=%-8.4f MAE=%-7.2f MAE/σ=%-6.3f CI80=%-5.1f%%║%n",
                name, ok ? "PASS" : "FAIL", maxDiff, mae, maeRatio, ci80));
            if (ok) pass++; else fail++;
        }

        // ── Test 2: Singapore ──
        {
            String name = "新加坡热带";
            String file = "singapore_temperature.model";
            int trainDays = 1826, testDays = 60;
            double noiseStd = 1.0;

            List<DataPoint> trainData = genSingapore(trainDays);
            List<DataPoint> allData = genSingapore(trainDays + testDays);
            double[] futureT = new double[testDays];
            double[] actualY = new double[testDays];
            for (int i = 0; i < testDays; i++) {
                futureT[i] = trainDays + i;
                actualY[i] = allData.get(trainDays + i).getValue();
            }

            ProphetConfig config = makeSingaporeConfig();
            ProphetModel fresh = new ProphetModel(config);
            fresh.fit(trainData);
            ProphetModel.ForecastResult freshResult = fresh.predict(futureT);

            ProphetModel loaded = ModelSerializer.load("models/" + file);
            ProphetModel.ForecastResult loadResult = loaded.predict(futureT);

            double maxDiff = 0;
            for (int i = 0; i < testDays; i++) {
                double diff = Math.abs(freshResult.yhat[i] - loadResult.yhat[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            double mae = 0, rmse = 0; int ciCount = 0;
            for (int i = 0; i < testDays; i++) {
                double err = Math.abs(loadResult.yhat[i] - actualY[i]);
                mae += err; rmse += err * err;
                if (actualY[i] >= loadResult.yhatLower[i] && actualY[i] <= loadResult.yhatUpper[i]) ciCount++;
            }
            mae /= testDays; rmse = Math.sqrt(rmse / testDays);
            double ci80 = ciCount * 100.0 / testDays;
            double maeRatio = mae / noiseStd;
            boolean ok = maxDiff < 1.0 && maeRatio < 1.5;

            printResult(name, ok, maxDiff, mae, rmse, maeRatio, ci80, maxDiff < 1.0, maeRatio < 1.5);
            summary.append(String.format("║ %-22s  %s  maxDiff=%-8.4f MAE=%-7.2f MAE/σ=%-6.3f CI80=%-5.1f%%║%n",
                name, ok ? "PASS" : "FAIL", maxDiff, mae, maeRatio, ci80));
            if (ok) pass++; else fail++;
        }

        // ── Test 3: Arctic ──
        {
            String name = "北极极地";
            String file = "arctic_temperature.model";
            int trainDays = 1826, testDays = 60;
            double noiseStd = 5.0;

            List<DataPoint> trainData = genArctic(trainDays);
            List<DataPoint> allData = genArctic(trainDays + testDays);
            double[] futureT = new double[testDays];
            double[] actualY = new double[testDays];
            for (int i = 0; i < testDays; i++) {
                futureT[i] = trainDays + i;
                actualY[i] = allData.get(trainDays + i).getValue();
            }

            ProphetConfig config = makeArcticConfig();
            ProphetModel fresh = new ProphetModel(config);
            fresh.fit(trainData);
            ProphetModel.ForecastResult freshResult = fresh.predict(futureT);

            ProphetModel loaded = ModelSerializer.load("models/" + file);
            ProphetModel.ForecastResult loadResult = loaded.predict(futureT);

            double maxDiff = 0;
            for (int i = 0; i < testDays; i++) {
                double diff = Math.abs(freshResult.yhat[i] - loadResult.yhat[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            double mae = 0, rmse = 0; int ciCount = 0;
            for (int i = 0; i < testDays; i++) {
                double err = Math.abs(loadResult.yhat[i] - actualY[i]);
                mae += err; rmse += err * err;
                if (actualY[i] >= loadResult.yhatLower[i] && actualY[i] <= loadResult.yhatUpper[i]) ciCount++;
            }
            mae /= testDays; rmse = Math.sqrt(rmse / testDays);
            double ci80 = ciCount * 100.0 / testDays;
            double maeRatio = mae / noiseStd;
            boolean ok = maxDiff < 1.0 && maeRatio < 1.5;

            printResult(name, ok, maxDiff, mae, rmse, maeRatio, ci80, maxDiff < 1.0, maeRatio < 1.5);
            summary.append(String.format("║ %-22s  %s  maxDiff=%-8.4f MAE=%-7.2f MAE/σ=%-6.3f CI80=%-5.1f%%║%n",
                name, ok ? "PASS" : "FAIL", maxDiff, mae, maeRatio, ci80));
            if (ok) pass++; else fail++;
        }

        // ── Test 4: Warming Trend ──
        {
            String name = "全球变暖趋势";
            String file = "warming_trend.model";
            int trainDays = 3652, testDays = 60;
            double noiseStd = 3.0;

            List<DataPoint> trainData = genWarming(trainDays);
            List<DataPoint> allData = genWarming(trainDays + testDays);
            double[] futureT = new double[testDays];
            double[] actualY = new double[testDays];
            for (int i = 0; i < testDays; i++) {
                futureT[i] = trainDays + i;
                actualY[i] = allData.get(trainDays + i).getValue();
            }

            ProphetConfig config = makeWarmingConfig();
            ProphetModel fresh = new ProphetModel(config);
            fresh.fit(trainData);
            ProphetModel.ForecastResult freshResult = fresh.predict(futureT);

            ProphetModel loaded = ModelSerializer.load("models/" + file);
            ProphetModel.ForecastResult loadResult = loaded.predict(futureT);

            double maxDiff = 0;
            for (int i = 0; i < testDays; i++) {
                double diff = Math.abs(freshResult.yhat[i] - loadResult.yhat[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            double mae = 0, rmse = 0; int ciCount = 0;
            for (int i = 0; i < testDays; i++) {
                double err = Math.abs(loadResult.yhat[i] - actualY[i]);
                mae += err; rmse += err * err;
                if (actualY[i] >= loadResult.yhatLower[i] && actualY[i] <= loadResult.yhatUpper[i]) ciCount++;
            }
            mae /= testDays; rmse = Math.sqrt(rmse / testDays);
            double ci80 = ciCount * 100.0 / testDays;
            double maeRatio = mae / noiseStd;
            boolean ok = maxDiff < 1.0 && maeRatio < 1.5;

            printResult(name, ok, maxDiff, mae, rmse, maeRatio, ci80, maxDiff < 1.0, maeRatio < 1.5);
            summary.append(String.format("║ %-22s  %s  maxDiff=%-8.4f MAE=%-7.2f MAE/σ=%-6.3f CI80=%-5.1f%%║%n",
                name, ok ? "PASS" : "FAIL", maxDiff, mae, maeRatio, ci80));
            if (ok) pass++; else fail++;
        }

        // ── Test 5: Multiplicative ──
        {
            String name = "乘法季节性";
            String file = "multiplicative_seasonal.model";
            int trainDays = 1826, testDays = 60;
            double noiseStd = 5.0;

            List<DataPoint> trainData = genMultiplicative(trainDays);
            List<DataPoint> allData = genMultiplicative(trainDays + testDays);
            double[] futureT = new double[testDays];
            double[] actualY = new double[testDays];
            for (int i = 0; i < testDays; i++) {
                futureT[i] = trainDays + i;
                actualY[i] = allData.get(trainDays + i).getValue();
            }

            ProphetConfig config = makeMultiplicativeConfig();
            ProphetModel fresh = new ProphetModel(config);
            fresh.fit(trainData);
            ProphetModel.ForecastResult freshResult = fresh.predict(futureT);

            ProphetModel loaded = ModelSerializer.load("models/" + file);
            ProphetModel.ForecastResult loadResult = loaded.predict(futureT);

            double maxDiff = 0;
            for (int i = 0; i < testDays; i++) {
                double diff = Math.abs(freshResult.yhat[i] - loadResult.yhat[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            double mae = 0, rmse = 0; int ciCount = 0;
            for (int i = 0; i < testDays; i++) {
                double err = Math.abs(loadResult.yhat[i] - actualY[i]);
                mae += err; rmse += err * err;
                if (actualY[i] >= loadResult.yhatLower[i] && actualY[i] <= loadResult.yhatUpper[i]) ciCount++;
            }
            mae /= testDays; rmse = Math.sqrt(rmse / testDays);
            double ci80 = ciCount * 100.0 / testDays;
            double maeRatio = mae / noiseStd;
            boolean ok = maxDiff < 1.0 && maeRatio < 1.5;

            printResult(name, ok, maxDiff, mae, rmse, maeRatio, ci80, maxDiff < 1.0, maeRatio < 1.5);
            summary.append(String.format("║ %-22s  %s  maxDiff=%-8.4f MAE=%-7.2f MAE/σ=%-6.3f CI80=%-5.1f%%║%n",
                name, ok ? "PASS" : "FAIL", maxDiff, mae, maeRatio, ci80));
            if (ok) pass++; else fail++;
        }

        // ── Test 6: Mixed Amp Growth ──
        {
            String name = "振幅增长MIXED";
            String file = "mixed_amp_growth.model";
            int trainDays = 2557, testDays = 60;
            double noiseStd = 3.0;

            List<DataPoint> trainData = genMixedAmp(trainDays);
            List<DataPoint> allData = genMixedAmp(trainDays + testDays);
            double[] futureT = new double[testDays];
            double[] actualY = new double[testDays];
            for (int i = 0; i < testDays; i++) {
                futureT[i] = trainDays + i;
                actualY[i] = allData.get(trainDays + i).getValue();
            }

            ProphetConfig config = makeMixedAmpConfig();
            ProphetModel fresh = new ProphetModel(config);
            fresh.fit(trainData);
            ProphetModel.ForecastResult freshResult = fresh.predict(futureT);

            ProphetModel loaded = ModelSerializer.load("models/" + file);
            ProphetModel.ForecastResult loadResult = loaded.predict(futureT);

            double maxDiff = 0;
            for (int i = 0; i < testDays; i++) {
                double diff = Math.abs(freshResult.yhat[i] - loadResult.yhat[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            double mae = 0, rmse = 0; int ciCount = 0;
            for (int i = 0; i < testDays; i++) {
                double err = Math.abs(loadResult.yhat[i] - actualY[i]);
                mae += err; rmse += err * err;
                if (actualY[i] >= loadResult.yhatLower[i] && actualY[i] <= loadResult.yhatUpper[i]) ciCount++;
            }
            mae /= testDays; rmse = Math.sqrt(rmse / testDays);
            double ci80 = ciCount * 100.0 / testDays;
            double maeRatio = mae / noiseStd;
            boolean ok = maxDiff < 1.0 && maeRatio < 1.5;

            printResult(name, ok, maxDiff, mae, rmse, maeRatio, ci80, maxDiff < 1.0, maeRatio < 1.5);
            summary.append(String.format("║ %-22s  %s  maxDiff=%-8.4f MAE=%-7.2f MAE/σ=%-6.3f CI80=%-5.1f%%║%n",
                name, ok ? "PASS" : "FAIL", maxDiff, mae, maeRatio, ci80));
            if (ok) pass++; else fail++;
        }

        // ── Test 7: Logistic ──
        {
            String name = "Logistic饱和";
            String file = "logistic_growth.model";
            int trainDays = 1826, testDays = 60;
            double noiseStd = 2.0;

            List<DataPoint> trainData = genLogistic(trainDays);
            List<DataPoint> allData = genLogistic(trainDays + testDays);
            double[] futureT = new double[testDays];
            double[] actualY = new double[testDays];
            for (int i = 0; i < testDays; i++) {
                futureT[i] = trainDays + i;
                actualY[i] = allData.get(trainDays + i).getValue();
            }

            ProphetConfig config = makeLogisticConfig();
            ProphetModel fresh = new ProphetModel(config);
            fresh.fit(trainData);
            ProphetModel.ForecastResult freshResult = fresh.predict(futureT);

            ProphetModel loaded = ModelSerializer.load("models/" + file);
            ProphetModel.ForecastResult loadResult = loaded.predict(futureT);

            double maxDiff = 0;
            for (int i = 0; i < testDays; i++) {
                double diff = Math.abs(freshResult.yhat[i] - loadResult.yhat[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            double mae = 0, rmse = 0; int ciCount = 0;
            for (int i = 0; i < testDays; i++) {
                double err = Math.abs(loadResult.yhat[i] - actualY[i]);
                mae += err; rmse += err * err;
                if (actualY[i] >= loadResult.yhatLower[i] && actualY[i] <= loadResult.yhatUpper[i]) ciCount++;
            }
            mae /= testDays; rmse = Math.sqrt(rmse / testDays);
            double ci80 = ciCount * 100.0 / testDays;
            double maeRatio = mae / noiseStd;
            boolean ok = maxDiff < 1.0 && maeRatio < 1.5;

            printResult(name, ok, maxDiff, mae, rmse, maeRatio, ci80, maxDiff < 1.0, maeRatio < 1.5);
            summary.append(String.format("║ %-22s  %s  maxDiff=%-8.4f MAE=%-7.2f MAE/σ=%-6.3f CI80=%-5.1f%%║%n",
                name, ok ? "PASS" : "FAIL", maxDiff, mae, maeRatio, ci80));
            if (ok) pass++; else fail++;
        }

        // ── Test 8: Weekly + Yearly ──
        {
            String name = "周+年双周期";
            String file = "weekly_yearly.model";
            int trainDays = 1826, testDays = 60;
            double noiseStd = 10.0;

            List<DataPoint> trainData = genWeeklyYearly(trainDays);
            List<DataPoint> allData = genWeeklyYearly(trainDays + testDays);
            double[] futureT = new double[testDays];
            double[] actualY = new double[testDays];
            for (int i = 0; i < testDays; i++) {
                futureT[i] = trainDays + i;
                actualY[i] = allData.get(trainDays + i).getValue();
            }

            ProphetConfig config = makeWeeklyYearlyConfig();
            ProphetModel fresh = new ProphetModel(config);
            fresh.fit(trainData);
            ProphetModel.ForecastResult freshResult = fresh.predict(futureT);

            ProphetModel loaded = ModelSerializer.load("models/" + file);
            ProphetModel.ForecastResult loadResult = loaded.predict(futureT);

            double maxDiff = 0;
            for (int i = 0; i < testDays; i++) {
                double diff = Math.abs(freshResult.yhat[i] - loadResult.yhat[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            double mae = 0, rmse = 0; int ciCount = 0;
            for (int i = 0; i < testDays; i++) {
                double err = Math.abs(loadResult.yhat[i] - actualY[i]);
                mae += err; rmse += err * err;
                if (actualY[i] >= loadResult.yhatLower[i] && actualY[i] <= loadResult.yhatUpper[i]) ciCount++;
            }
            mae /= testDays; rmse = Math.sqrt(rmse / testDays);
            double ci80 = ciCount * 100.0 / testDays;
            double maeRatio = mae / noiseStd;
            boolean ok = maxDiff < 1.0 && maeRatio < 1.5;

            printResult(name, ok, maxDiff, mae, rmse, maeRatio, ci80, maxDiff < 1.0, maeRatio < 1.5);
            summary.append(String.format("║ %-22s  %s  maxDiff=%-8.4f MAE=%-7.2f MAE/σ=%-6.3f CI80=%-5.1f%%║%n",
                name, ok ? "PASS" : "FAIL", maxDiff, mae, maeRatio, ci80));
            if (ok) pass++; else fail++;
        }

        // ── Summary ──
        System.out.println("\n╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║                     CONSISTENCY & ACCURACY                     ║");
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        System.out.print(summary);
        System.out.println("╠══════════════════════════════════════════════════════════════════╣");
        System.out.printf("║ Total: %d PASS / %d FAIL / %d total                             ║%n", pass, fail, pass + fail);
        System.out.println("╚══════════════════════════════════════════════════════════════════╝");
    }

    static void printResult(String name, boolean ok, double maxDiff,
                             double mae, double rmse, double maeRatio,
                             double ci80, boolean consistency, boolean accuracy) {
        System.out.println("─── " + name + " ───");
        System.out.printf("  Consistency (fresh vs loaded): maxDiff=%.6f → %s%n",
            maxDiff, consistency ? "OK" : "MISMATCH");
        System.out.printf("  Accuracy: MAE=%.2f, RMSE=%.2f, MAE/σ=%.3f, CI80=%.1f%% → %s%n",
            mae, rmse, maeRatio, ci80, accuracy ? "OK" : "POOR");
        System.out.printf("  Overall: %s%n%n", ok ? "PASS" : "FAIL");
    }

    // ─── Data generators (identical to TrainModels) ───

    static List<DataPoint> genBeijing(int n) {
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED);
        for (int day = 0; day < n; day++) {
            double y = 12.0 + 20.0 * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + rng.nextGaussian() * 3.0;
            data.add(new DataPoint(day, y));
        }
        return data;
    }

    static ProphetConfig makeBeijingConfig() {
        ProphetConfig c = new ProphetConfig();
        c.growth = ProphetConfig.GrowthType.LINEAR;
        c.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        c.nChangepoints = 25; c.changepointPriorScale = 0.05;
        c.yearlyFourierOrder = 10; c.yearlySeasonality = true; c.weeklySeasonality = false;
        c.mcmcSamples = 0; c.verbose = false;
        return c;
    }

    static List<DataPoint> genSingapore(int n) {
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 1);
        for (int day = 0; day < n; day++) {
            double y = 27.0 + 2.0 * Math.sin(2 * Math.PI * (day - 30) / 365.25)
                     + 0.5 * Math.sin(2 * Math.PI * day / 365.25 * 2)
                     + rng.nextGaussian() * 1.0;
            data.add(new DataPoint(day, y));
        }
        return data;
    }

    static ProphetConfig makeSingaporeConfig() {
        ProphetConfig c = new ProphetConfig();
        c.growth = ProphetConfig.GrowthType.LINEAR;
        c.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        c.nChangepoints = 25; c.changepointPriorScale = 0.05;
        c.yearlyFourierOrder = 10; c.yearlySeasonality = true; c.weeklySeasonality = false;
        c.mcmcSamples = 0; c.verbose = false;
        return c;
    }

    static List<DataPoint> genArctic(int n) {
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 2);
        for (int day = 0; day < n; day++) {
            double y = -20.0 + 25.0 * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + rng.nextGaussian() * 5.0;
            data.add(new DataPoint(day, y));
        }
        return data;
    }

    static ProphetConfig makeArcticConfig() {
        ProphetConfig c = new ProphetConfig();
        c.growth = ProphetConfig.GrowthType.LINEAR;
        c.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        c.nChangepoints = 25; c.changepointPriorScale = 0.05;
        c.yearlyFourierOrder = 10; c.yearlySeasonality = true; c.weeklySeasonality = false;
        c.mcmcSamples = 0; c.verbose = false;
        return c;
    }

    static List<DataPoint> genWarming(int n) {
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 3);
        for (int day = 0; day < n; day++) {
            double y = 14.0 + 0.02 * day / 365.25
                     + 18.0 * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + rng.nextGaussian() * 3.0;
            data.add(new DataPoint(day, y));
        }
        return data;
    }

    static ProphetConfig makeWarmingConfig() {
        ProphetConfig c = new ProphetConfig();
        c.growth = ProphetConfig.GrowthType.LINEAR;
        c.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        c.nChangepoints = 25; c.changepointPriorScale = 0.05;
        c.yearlyFourierOrder = 10; c.yearlySeasonality = true; c.weeklySeasonality = false;
        c.mcmcSamples = 0; c.verbose = false;
        return c;
    }

    static List<DataPoint> genMultiplicative(int n) {
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 4);
        for (int day = 0; day < n; day++) {
            double trend = 100 + 10 * day / 365.25;
            double y = trend * (1 + 0.3 * Math.sin(2 * Math.PI * (day - 80) / 365.25))
                     + rng.nextGaussian() * 5.0;
            data.add(new DataPoint(day, y));
        }
        return data;
    }

    static ProphetConfig makeMultiplicativeConfig() {
        ProphetConfig c = new ProphetConfig();
        c.growth = ProphetConfig.GrowthType.LINEAR;
        c.seasonalityMode = ProphetConfig.SeasonalityMode.MULTIPLICATIVE;
        c.nChangepoints = 25; c.changepointPriorScale = 0.05;
        c.yearlyFourierOrder = 10; c.yearlySeasonality = true; c.weeklySeasonality = false;
        c.mcmcSamples = 0; c.verbose = false;
        return c;
    }

    static List<DataPoint> genMixedAmp(int n) {
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 5);
        for (int day = 0; day < n; day++) {
            double trend = 15.0 + 5.0 * day / 365.25;
            double ampGrowth = 1.0 + 0.5 * day / 2557.0;
            double y = trend
                     + 10.0 * ampGrowth * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + rng.nextGaussian() * 3.0;
            data.add(new DataPoint(day, y));
        }
        return data;
    }

    static ProphetConfig makeMixedAmpConfig() {
        ProphetConfig c = new ProphetConfig();
        c.growth = ProphetConfig.GrowthType.LINEAR;
        c.seasonalityMode = ProphetConfig.SeasonalityMode.MIXED;
        c.nChangepoints = 25; c.changepointPriorScale = 0.05;
        c.yearlyFourierOrder = 10; c.yearlySeasonality = true; c.weeklySeasonality = false;
        c.mcmcSamples = 0; c.verbose = false;
        return c;
    }

    static List<DataPoint> genLogistic(int n) {
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 6);
        double cap = 100.0;
        for (int day = 0; day < n; day++) {
            double k = 0.003, m = 500.0;
            double logistic = cap / (1 + Math.exp(-k * (day - m)));
            double y = logistic + 5.0 * Math.sin(2 * Math.PI * day / 365.25)
                     + rng.nextGaussian() * 2.0;
            y = Math.max(y, 0.1);
            data.add(new DataPoint(day, y));
        }
        return data;
    }

    static ProphetConfig makeLogisticConfig() {
        ProphetConfig c = new ProphetConfig();
        c.growth = ProphetConfig.GrowthType.LOGISTIC;
        c.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        c.cap = 100.0; c.floor = 0.0;
        c.nChangepoints = 25; c.changepointPriorScale = 0.05;
        c.yearlyFourierOrder = 10; c.yearlySeasonality = true; c.weeklySeasonality = false;
        c.mcmcSamples = 0; c.verbose = false;
        return c;
    }

    static List<DataPoint> genWeeklyYearly(int n) {
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 7);
        for (int day = 0; day < n; day++) {
            double y = 500.0
                     + 80.0 * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + 40.0 * Math.sin(2 * Math.PI * day / 7.0)
                     + 0.01 * day
                     + rng.nextGaussian() * 10.0;
            data.add(new DataPoint(day, y));
        }
        return data;
    }

    static ProphetConfig makeWeeklyYearlyConfig() {
        ProphetConfig c = new ProphetConfig();
        c.growth = ProphetConfig.GrowthType.LINEAR;
        c.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        c.nChangepoints = 25; c.changepointPriorScale = 0.05;
        c.yearlyFourierOrder = 10; c.weeklyFourierOrder = 3;
        c.yearlySeasonality = true; c.weeklySeasonality = true;
        c.mcmcSamples = 0; c.verbose = false;
        return c;
    }
}
