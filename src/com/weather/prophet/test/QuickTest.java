package com.weather.prophet.test;

import com.weather.prophet.core.*;
import com.weather.prophet.data.DataPoint;
import java.util.*;

/** Quick single-scenario test to verify L-BFGS convergence */
public class QuickTest {
    public static void main(String[] args) {
        Random rng = new Random(42);
        // Beijing-style temperature
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < 730; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = 13 + 20 * Math.cos(yearPhase - Math.PI) + rng.nextGaussian() * 3;
            data.add(new DataPoint(t, y));
        }

        ProphetConfig config = new ProphetConfig();
        config.growth = ProphetConfig.GrowthType.LINEAR;
        config.nChangepoints = 25;
        config.changepointPriorScale = 0.05;
        config.yearlySeasonality = true;
        config.yearlyFourierOrder = 10;
        config.weeklySeasonality = false;
        config.mcmcSamples = 0;
        config.uncertaintySamples = 100;
        config.verbose = true;

        int testDays = 60;
        List<DataPoint> trainData = data.subList(0, data.size() - testDays);
        List<DataPoint> testData = data.subList(data.size() - testDays, data.size());

        ProphetModel model = new ProphetModel(config);
        model.fit(trainData);

        double[] testT = new double[testDays];
        double[] testY = new double[testDays];
        for (int i = 0; i < testDays; i++) {
            testT[i] = testData.get(i).getTimestamp();
            testY[i] = testData.get(i).getValue();
        }

        ProphetModel.ForecastResult forecast = model.predict(testT);

        // Compute metrics
        double ssRes = 0, ssTot = 0, sumAbsErr = 0;
        double yMean = 0;
        for (double v : testY) yMean += v;
        yMean /= testDays;

        System.out.println("\n=== Prediction Results ===");
        for (int i = 0; i < testDays; i++) {
            double err = forecast.yhat[i] - testY[i];
            ssRes += err * err;
            ssTot += (testY[i] - yMean) * (testY[i] - yMean);
            sumAbsErr += Math.abs(err);
            if (i < 10 || i > testDays - 5) {
                System.out.printf("  t=%3d: actual=%6.1f  pred=%6.1f  lower=%6.1f  upper=%6.1f%n",
                    i, testY[i], forecast.yhat[i], forecast.yhatLower[i], forecast.yhatUpper[i]);
            }
        }

        double r2 = 1 - ssRes / ssTot;
        double mae = sumAbsErr / testDays;
        System.out.printf("\nR²=%.4f  MAE=%.2f°C  RMSE=%.2f°C%n", r2, mae, Math.sqrt(ssRes/testDays));
    }
}
