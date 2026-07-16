package com.weather.prophet.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Prophet configuration — mirrors Prophet's Python constructor parameters exactly.
 * Reference: facebook/prophet python/prophet/forecaster.py __init__
 */
public class ProphetConfig {

    public enum GrowthType { LINEAR, LOGISTIC, FLAT }
    public enum SeasonalityMode { ADDITIVE, MULTIPLICATIVE, MIXED }
    public enum Scaling { ABSMAX, MINMAX }

    // Growth
    public GrowthType growth = GrowthType.LINEAR;
    public int nChangepoints = 25;
    public double changepointRange = 0.8;
    public double changepointPriorScale = 0.05;   // tau for Laplace prior on delta

    // Seasonality
    public SeasonalityMode seasonalityMode = SeasonalityMode.ADDITIVE;
    public double seasonalityPriorScale = 10.0;   // sigma_beta for all seasonal features
    public double yearlySeasonalityPriorScale = 10.0;
    public double weeklySeasonalityPriorScale = 10.0;
    public double dailySeasonalityPriorScale = 10.0;
    public int yearlyFourierOrder = 10;
    public int weeklyFourierOrder = 3;
    public int dailyFourierOrder = 4;
    public boolean yearlySeasonality = true;
    public boolean weeklySeasonality = false;
    public boolean dailySeasonality = false;

    // Holidays
    public double holidaysPriorScale = 10.0;       // sigma_kappa
    public SeasonalityMode holidaysMode = null;    // null = same as seasonalityMode

    // MCMC
    public int mcmcSamples = 0;                     // 0 = MAP only (Prophet default)
    public int mcmcWarmup = 500;
    public int mcmcChains = 4;
    public boolean useNUTS = true;

    // Uncertainty
    public int uncertaintySamples = 1000;
    public double sigmaObsPriorScale = 0.5;         // Normal(0, scale) for sigma_obs (NOT HalfCauchy)

    // Optimization
    public int lbfgsMaxIter = 10000;
    public double lbfgsGradTol = 1e-8;
    public boolean verbose = true;

    // Scaling
    public Scaling scaling = Scaling.ABSMAX;        // Prophet default: 'absmax'

    // Logistic growth params
    public double cap = Double.MAX_VALUE;           // carrying capacity C
    public double floor = 0.0;                      // saturating minimum (Prophet default 0)

    // Annual period (365.25 to account for leap years, matching Prophet)
    public double yearlyPeriod = 365.25;
    public double weeklyPeriod = 7.0;
    public double dailyPeriod = 1.0;

    // Holiday definitions
    public List<HolidaySpec> holidays = new ArrayList<>();

    public static class HolidaySpec {
        public final String name;
        public final double timestamp;
        public final int windowBefore;
        public final int windowAfter;

        public HolidaySpec(String name, double timestamp, int windowBefore, int windowAfter) {
            this.name = name;
            this.timestamp = timestamp;
            this.windowBefore = windowBefore;
            this.windowAfter = windowAfter;
        }

        public HolidaySpec(String name, double timestamp) {
            this(name, timestamp, 1, 1);
        }
    }

    public ProphetConfig addHoliday(String name, double timestamp, int wb, int wa) {
        holidays.add(new HolidaySpec(name, timestamp, wb, wa));
        return this;
    }

    public ProphetConfig addHoliday(String name, double timestamp) {
        holidays.add(new HolidaySpec(name, timestamp));
        return this;
    }

    /** Effective holidays mode: if null, use seasonalityMode */
    public SeasonalityMode getHolidaysMode() {
        return holidaysMode != null ? holidaysMode : seasonalityMode;
    }
}
