package com.weather.prophet.core;

import com.weather.prophet.data.DataPoint;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * Serialize / deserialize a trained ProphetModel to a portable JSON-like text file.
 * The weights file contains all parameters needed for prediction without retraining.
 */
public class ModelSerializer {

    // ===================== Save =====================

    public static void save(ProphetModel model, String path) throws IOException {
        try (BufferedWriter w = new BufferedWriter(new FileWriter(path))) {
            w.write("# WeatherProphet Model Weights"); w.newLine();
            w.write("# Generated: " + new Date()); w.newLine();
            w.write("---"); w.newLine();

            // Config
            ProphetConfig c = model.getConfig();
            writeSection(w, "config");
            writeKV(w, "growth", c.growth.name());
            writeKV(w, "seasonalityMode", c.seasonalityMode.name());
            writeKV(w, "scaling", c.scaling.name());
            writeKV(w, "nChangepoints", c.nChangepoints);
            writeKV(w, "changepointPriorScale", c.changepointPriorScale);
            writeKV(w, "changepointRange", c.changepointRange);
            writeKV(w, "yearlyFourierOrder", c.yearlyFourierOrder);
            writeKV(w, "weeklyFourierOrder", c.weeklyFourierOrder);
            writeKV(w, "dailyFourierOrder", c.dailyFourierOrder);
            writeKV(w, "yearlySeasonality", c.yearlySeasonality);
            writeKV(w, "weeklySeasonality", c.weeklySeasonality);
            writeKV(w, "dailySeasonality", c.dailySeasonality);
            writeKV(w, "yearlySeasonalityPriorScale", c.yearlySeasonalityPriorScale);
            writeKV(w, "weeklySeasonalityPriorScale", c.weeklySeasonalityPriorScale);
            writeKV(w, "dailySeasonalityPriorScale", c.dailySeasonalityPriorScale);
            writeKV(w, "holidaysPriorScale", c.holidaysPriorScale);
            writeKV(w, "sigmaObsPriorScale", c.sigmaObsPriorScale);
            writeKV(w, "yearlyPeriod", c.yearlyPeriod);
            writeKV(w, "weeklyPeriod", c.weeklyPeriod);
            writeKV(w, "dailyPeriod", c.dailyPeriod);
            writeKV(w, "mcmcSamples", c.mcmcSamples);

            // Scaling params
            writeSection(w, "scaling_params");
            writeKV(w, "yScale", model.getYScale());
            writeKV(w, "yOffset", model.getYOffset());
            writeKV(w, "tMin", model.getTMin());
            writeKV(w, "tScale", model.getTScale());
            writeKV(w, "trainingMaxT", model.getTrainingMaxT());
            writeKV(w, "growthCode", model.getGrowthCode());

            // Changepoints
            writeSection(w, "changepoints");
            writeArray(w, model.getChangepoints());

            // Fitted parameters
            writeSection(w, "params");
            writeKV(w, "k", model.getK());
            writeKV(w, "m", model.getM());
            writeKV(w, "sigmaObs", model.getSigmaObs());
            writeArray(w, "delta", model.getDelta());
            writeArray(w, "beta", model.getBeta());

            // Indicator vectors
            writeSection(w, "indicators");
            writeArray(w, "s_a", model.getSA());
            writeArray(w, "s_m", model.getSM());
            writeArray(w, "sigmas", model.getSigmas());

            // Cap (for logistic)
            if (model.getCapScaled() != null) {
                writeArray(w, "capScaled", model.getCapScaled());
            }

            // Training stats
            writeSection(w, "training_stats");
            writeKV(w, "trainSize", model.getTrainT().length);
            writeArray(w, "trainY_sample", sampleArray(model.getTrainY(), 50));

            w.flush();
        }
    }

    // ===================== Load =====================

    public static ProphetModel load(String path) throws IOException {
        Map<String, String> kv = new LinkedHashMap<>();
        Map<String, double[]> arrays = new LinkedHashMap<>();

        try (BufferedReader r = new BufferedReader(new FileReader(path))) {
            String section = "";
            String line;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.startsWith("#") || line.isEmpty()) continue;
                if (line.equals("---")) continue;
                if (line.startsWith("===") && line.endsWith("===")) {
                    section = line.substring(3, line.length() - 3).trim();
                    continue;
                }
                if (line.contains(":") && !line.startsWith("[")) {
                    String[] parts = line.split(":", 2);
                    kv.put(section + "." + parts[0].trim(), parts[1].trim());
                }
                if (line.startsWith("[")) {
                    String name = line.substring(0, line.indexOf(":"));
                    // Actually parse: name:[v1,v2,...]
                    int colonIdx = line.indexOf(':');
                    if (colonIdx > 0) {
                        String arrName = line.substring(0, colonIdx).trim();
                        String arrStr = line.substring(colonIdx + 1).trim();
                        arrays.put(section + "." + arrName, parseArray(arrStr));
                    }
                }
            }
        }

        // Rebuild config
        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.valueOf(kv.get("config.growth"));
        config.seasonalityMode = ProphetConfig.SeasonalityMode.valueOf(kv.get("config.seasonalityMode"));
        config.scaling = ProphetConfig.Scaling.valueOf(kv.get("config.scaling"));
        config.nChangepoints = Integer.parseInt(kv.get("config.nChangepoints"));
        config.changepointPriorScale = Double.parseDouble(kv.get("config.changepointPriorScale"));
        config.changepointRange = Double.parseDouble(kv.get("config.changepointRange"));
        config.yearlyFourierOrder = Integer.parseInt(kv.get("config.yearlyFourierOrder"));
        config.weeklyFourierOrder = Integer.parseInt(kv.getOrDefault("config.weeklyFourierOrder", "3"));
        config.dailyFourierOrder = Integer.parseInt(kv.getOrDefault("config.dailyFourierOrder", "4"));
        config.yearlySeasonality = Boolean.parseBoolean(kv.getOrDefault("config.yearlySeasonality", "true"));
        config.weeklySeasonality = Boolean.parseBoolean(kv.getOrDefault("config.weeklySeasonality", "false"));
        config.dailySeasonality = Boolean.parseBoolean(kv.getOrDefault("config.dailySeasonality", "false"));
        config.yearlySeasonalityPriorScale = Double.parseDouble(kv.getOrDefault("config.yearlySeasonalityPriorScale", "10.0"));
        config.weeklySeasonalityPriorScale = Double.parseDouble(kv.getOrDefault("config.weeklySeasonalityPriorScale", "10.0"));
        config.dailySeasonalityPriorScale = Double.parseDouble(kv.getOrDefault("config.dailySeasonalityPriorScale", "10.0"));
        config.holidaysPriorScale = Double.parseDouble(kv.getOrDefault("config.holidaysPriorScale", "10.0"));
        config.sigmaObsPriorScale = Double.parseDouble(kv.getOrDefault("config.sigmaObsPriorScale", "0.5"));
        config.yearlyPeriod = Double.parseDouble(kv.getOrDefault("config.yearlyPeriod", "365.25"));
        config.weeklyPeriod = Double.parseDouble(kv.getOrDefault("config.weeklyPeriod", "7.0"));
        config.dailyPeriod = Double.parseDouble(kv.getOrDefault("config.dailyPeriod", "1.0"));
        config.mcmcSamples = Integer.parseInt(kv.getOrDefault("config.mcmcSamples", "0"));
        config.verbose = false;

        // Create model and restore state
        ProphetModel model = new ProphetModel(config);

        // We need to set internal state via fit with dummy data, then override
        // Actually, let's create a minimal fit and then set params
        // Create dummy training data (1 year)
        List<DataPoint> dummyData = new ArrayList<>();
        double yScaleLoaded = Double.parseDouble(kv.get("scaling_params.yScale"));
        double yOffsetLoaded = Double.parseDouble(kv.get("scaling_params.yOffset"));
        double tMinLoaded = Double.parseDouble(kv.get("scaling_params.tMin"));
        double tScaleLoaded = Double.parseDouble(kv.get("scaling_params.tScale"));
        int trainSize = Integer.parseInt(kv.getOrDefault("scaling_params.trainSize", "365"));

        for (int i = 0; i < trainSize; i++) {
            double t = tMinLoaded + i;
            dummyData.add(new DataPoint(t, yOffsetLoaded + 10 * Math.sin(2 * Math.PI * i / 365.25)));
        }

        // Fit to initialize internal structures
        config.verbose = false;
        model.fit(dummyData);

        // Now override fitted parameters with saved ones
        model.setK(Double.parseDouble(kv.get("params.k")));
        model.setM(Double.parseDouble(kv.get("params.m")));
        model.setSigmaObs(Double.parseDouble(kv.get("params.sigmaObs")));
        model.setDelta(arrays.getOrDefault("params.delta", new double[0]));
        model.setBeta(arrays.getOrDefault("params.beta", new double[0]));

        return model;
    }

    // ===================== Helpers =====================

    private static void writeSection(BufferedWriter w, String name) throws IOException {
        w.write("===" + name + "==="); w.newLine();
    }

    private static void writeKV(BufferedWriter w, String key, Object val) throws IOException {
        w.write(key + ": " + val); w.newLine();
    }

    private static void writeArray(BufferedWriter w, double[] arr) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(String.format(Locale.US, "%.10g", arr[i]));
        }
        sb.append("]");
        w.newLine();
        w.write("array_len: " + arr.length); w.newLine();
        // Write in chunks of 10 per line for readability
        int cols = 10;
        for (int i = 0; i < arr.length; i += cols) {
            StringBuilder line = new StringBuilder();
            for (int j = i; j < Math.min(i + cols, arr.length); j++) {
                if (j > i) line.append(" ");
                line.append(String.format(Locale.US, "%.10g", arr[j]));
            }
            w.write(line.toString()); w.newLine();
        }
    }

    private static void writeArray(BufferedWriter w, String name, double[] arr) throws IOException {
        w.write(name + ": len=" + arr.length); w.newLine();
        int cols = 10;
        for (int i = 0; i < arr.length; i += cols) {
            StringBuilder line = new StringBuilder();
            for (int j = i; j < Math.min(i + cols, arr.length); j++) {
                if (j > i) line.append(" ");
                line.append(String.format(Locale.US, "%.10g", arr[j]));
            }
            w.write(line.toString()); w.newLine();
        }
    }

    private static double[] parseArray(String str) {
        str = str.trim();
        if (str.startsWith("[")) str = str.substring(1);
        if (str.endsWith("]")) str = str.substring(0, str.length() - 1);
        if (str.isEmpty()) return new double[0];
        String[] parts = str.split("[,\\s]+");
        double[] arr = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            arr[i] = Double.parseDouble(parts[i].trim());
        }
        return arr;
    }

    private static double[] sampleArray(double[] arr, int maxSamples) {
        if (arr.length <= maxSamples) return arr;
        double[] sample = new double[maxSamples];
        for (int i = 0; i < maxSamples; i++) {
            sample[i] = arr[i * arr.length / maxSamples];
        }
        return sample;
    }
}
