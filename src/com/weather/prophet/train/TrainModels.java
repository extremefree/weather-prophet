package com.weather.prophet.train;

import com.weather.prophet.core.ModelSerializer;
import com.weather.prophet.core.ProphetConfig;
import com.weather.prophet.core.ProphetModel;
import com.weather.prophet.data.DataPoint;

import java.io.*;
import java.util.*;

/**
 * Train Prophet models on realistic weather data and save weights.
 * Produces multiple model variants:
 *   1. beijing_temperature.model    — 4-season temperate climate (linear + additive)
 *   2. singapore_temperature.model  — tropical climate (linear + additive)
 *   3. arctic_temperature.model     — polar climate (linear + additive)
 *   4. warming_trend.model          — global warming trend (linear + additive)
 *   5. multiplicative_seasonal.model — multiplicative seasonality (linear + multiplicative)
 *   6. mixed_amp_growth.model       — amplitude growth (linear + mixed)
 *   7. logistic_growth.model        — logistic saturation (logistic + additive)
 *   8. weekly_yearly.model          — weekly+yearly dual seasonality
 */
public class TrainModels {

    static final String OUTPUT_DIR = "models";
    static final long SEED = 42;

    public static void main(String[] args) throws Exception {
        File dir = new File(OUTPUT_DIR);
        if (!dir.exists()) dir.mkdirs();

        System.out.println("========================================");
        System.out.println("  WeatherProphet Model Training Suite  ");
        System.out.println("========================================\n");

        // 1. Beijing — 4-season temperate
        trainBeijing();

        // 2. Singapore — tropical (low seasonality amplitude)
        trainSingapore();

        // 3. Arctic — extreme cold with mild summer
        trainArctic();

        // 4. Warming trend — upward linear trend + seasonality
        trainWarmingTrend();

        // 5. Multiplicative seasonality
        trainMultiplicative();

        // 6. Mixed amplitude growth
        trainMixedAmpGrowth();

        // 7. Logistic growth
        trainLogistic();

        // 8. Weekly + Yearly dual seasonality
        trainWeeklyYearly();

        System.out.println("\n========================================");
        System.out.println("  All models trained and saved!        ");
        System.out.println("========================================");
    }

    // ─── Data generators ───

    /**
     * Generate realistic Beijing temperature data.
     * Annual cycle: mean 12°C, amplitude ~20°C, noise σ=3
     * Training: 5 years (1826 days), daily resolution
     */
    static void trainBeijing() throws Exception {
        System.out.println("[1/8] Training Beijing temperature model...");
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED);
        for (int day = 0; day < 1826; day++) {
            double t = day;
            double y = 12.0 + 20.0 * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + rng.nextGaussian() * 3.0;
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlyFourierOrder = 10;
        config.yearlySeasonality = true;
        config.weeklySeasonality = false;
        config.mcmcSamples = 0;
        config.verbose = false;

        ProphetModel model = new ProphetModel(config);
        model.fit(data);

        String path = OUTPUT_DIR + "/beijing_temperature.model";
        ModelSerializer.save(model, path);
        System.out.println("  -> Saved: " + path + " (sigma=" + model.getSigmaObs() + ")");
    }

    /**
     * Singapore — tropical, annual amplitude ~2°C around 27°C
     */
    static void trainSingapore() throws Exception {
        System.out.println("[2/8] Training Singapore temperature model...");
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 1);
        for (int day = 0; day < 1826; day++) {
            double t = day;
            double y = 27.0 + 2.0 * Math.sin(2 * Math.PI * (day - 30) / 365.25)
                     + 0.5 * Math.sin(2 * Math.PI * day / 365.25 * 2)  // semi-annual
                     + rng.nextGaussian() * 1.0;
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlyFourierOrder = 10;
        config.yearlySeasonality = true;
        config.weeklySeasonality = false;
        config.mcmcSamples = 0;
        config.verbose = false;

        ProphetModel model = new ProphetModel(config);
        model.fit(data);

        String path = OUTPUT_DIR + "/singapore_temperature.model";
        ModelSerializer.save(model, path);
        System.out.println("  -> Saved: " + path + " (sigma=" + model.getSigmaObs() + ")");
    }

    /**
     * Arctic — mean -20°C, summer peak ~5°C, winter low ~-40°C
     */
    static void trainArctic() throws Exception {
        System.out.println("[3/8] Training Arctic temperature model...");
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 2);
        for (int day = 0; day < 1826; day++) {
            double t = day;
            double y = -20.0 + 25.0 * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + rng.nextGaussian() * 5.0;
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlyFourierOrder = 10;
        config.yearlySeasonality = true;
        config.weeklySeasonality = false;
        config.mcmcSamples = 0;
        config.verbose = false;

        ProphetModel model = new ProphetModel(config);
        model.fit(data);

        String path = OUTPUT_DIR + "/arctic_temperature.model";
        ModelSerializer.save(model, path);
        System.out.println("  -> Saved: " + path + " (sigma=" + model.getSigmaObs() + ")");
    }

    /**
     * Warming trend — +0.02°C/year linear trend + strong seasonality
     */
    static void trainWarmingTrend() throws Exception {
        System.out.println("[4/8] Training warming trend model...");
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 3);
        for (int day = 0; day < 3652; day++) {  // 10 years
            double t = day;
            double y = 14.0 + 0.02 * day / 365.25
                     + 18.0 * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + rng.nextGaussian() * 3.0;
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlyFourierOrder = 10;
        config.yearlySeasonality = true;
        config.weeklySeasonality = false;
        config.mcmcSamples = 0;
        config.verbose = false;

        ProphetModel model = new ProphetModel(config);
        model.fit(data);

        String path = OUTPUT_DIR + "/warming_trend.model";
        ModelSerializer.save(model, path);
        System.out.println("  -> Saved: " + path + " (sigma=" + model.getSigmaObs() + ")");
    }

    /**
     * Multiplicative seasonality — amplitude scales with level
     */
    static void trainMultiplicative() throws Exception {
        System.out.println("[5/8] Training multiplicative seasonality model...");
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 4);
        for (int day = 0; day < 1826; day++) {
            double t = day;
            double trend = 100 + 10 * day / 365.25;
            double y = trend * (1 + 0.3 * Math.sin(2 * Math.PI * (day - 80) / 365.25))
                     + rng.nextGaussian() * 5.0;
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.MULTIPLICATIVE;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlyFourierOrder = 10;
        config.yearlySeasonality = true;
        config.weeklySeasonality = false;
        config.mcmcSamples = 0;
        config.verbose = false;

        ProphetModel model = new ProphetModel(config);
        model.fit(data);

        String path = OUTPUT_DIR + "/multiplicative_seasonal.model";
        ModelSerializer.save(model, path);
        System.out.println("  -> Saved: " + path + " (sigma=" + model.getSigmaObs() + ")");
    }

    /**
     * Mixed amplitude growth — additive + multiplicative seasonality
     */
    static void trainMixedAmpGrowth() throws Exception {
        System.out.println("[6/8] Training mixed amplitude growth model...");
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 5);
        for (int day = 0; day < 2557; day++) {  // 7 years
            double t = day;
            double trend = 15.0 + 5.0 * day / 365.25;
            double ampGrowth = 1.0 + 0.5 * day / 2557.0;
            double y = trend
                     + 10.0 * ampGrowth * Math.sin(2 * Math.PI * (day - 80) / 365.25)
                     + rng.nextGaussian() * 3.0;
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.MIXED;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlyFourierOrder = 10;
        config.yearlySeasonality = true;
        config.weeklySeasonality = false;
        config.mcmcSamples = 0;
        config.verbose = false;

        ProphetModel model = new ProphetModel(config);
        model.fit(data);

        String path = OUTPUT_DIR + "/mixed_amp_growth.model";
        ModelSerializer.save(model, path);
        System.out.println("  -> Saved: " + path + " (sigma=" + model.getSigmaObs() + ")");
    }

    /**
     * Logistic growth — saturating capacity
     */
    static void trainLogistic() throws Exception {
        System.out.println("[7/8] Training logistic growth model...");
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 6);
        double cap = 100.0;
        for (int day = 0; day < 1826; day++) {
            double t = day;
            double k = 0.003;
            double m = 500.0;
            double logistic = cap / (1 + Math.exp(-k * (t - m)));
            double y = logistic + 5.0 * Math.sin(2 * Math.PI * day / 365.25)
                     + rng.nextGaussian() * 2.0;
            y = Math.max(y, 0.1);
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LOGISTIC;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        config.cap = cap;
        config.floor = 0.0;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlyFourierOrder = 10;
        config.yearlySeasonality = true;
        config.weeklySeasonality = false;
        config.mcmcSamples = 0;
        config.verbose = false;

        ProphetModel model = new ProphetModel(config);
        model.fit(data);

        String path = OUTPUT_DIR + "/logistic_growth.model";
        ModelSerializer.save(model, path);
        System.out.println("  -> Saved: " + path + " (sigma=" + model.getSigmaObs() + ")");
    }

    /**
     * Weekly + Yearly dual seasonality (e.g., electricity demand)
     */
    static void trainWeeklyYearly() throws Exception {
        System.out.println("[8/8] Training weekly+yearly dual seasonality model...");
        List<DataPoint> data = new ArrayList<>();
        Random rng = new Random(SEED + 7);
        for (int day = 0; day < 1826; day++) {
            double t = day;
            double y = 500.0
                     + 80.0 * Math.sin(2 * Math.PI * (day - 80) / 365.25)   // yearly
                     + 40.0 * Math.sin(2 * Math.PI * day / 7.0)              // weekly
                     + 0.01 * day                                            // slight trend
                     + rng.nextGaussian() * 10.0;
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlyFourierOrder = 10;
        config.weeklyFourierOrder = 3;
        config.yearlySeasonality = true;
        config.weeklySeasonality = true;
        config.mcmcSamples = 0;
        config.verbose = false;

        ProphetModel model = new ProphetModel(config);
        model.fit(data);

        String path = OUTPUT_DIR + "/weekly_yearly.model";
        ModelSerializer.save(model, path);
        System.out.println("  -> Saved: " + path + " (sigma=" + model.getSigmaObs() + ")");
    }
}
