package com.weather.prophet.core;

import java.util.ArrayList;
import java.util.List;

/**
 * Prophet configuration — mirrors Prophet's Python constructor parameters exactly.
 *
 * Key parameters (from facebook/prophet):
 * - growth: 'linear' or 'logistic'
 * - changepoints: custom changepoint dates (null = auto)
 * - n_changepoints: number of automatic changepoints (default 25)
 * - changepoint_range: proportion of history for changepoints (default 0.8)
 * - changepoint_prior_scale: Laplace scale tau (default 0.05)
 * - seasonality_prior_scale: Normal scale for beta (default 10)
 * - holidays_prior_scale: Normal scale for kappa (default 10)
 * - seasonality_mode: 'additive' or 'multiplicative'
 * - mcmc_samples: 0 = MAP only, >0 = MCMC samples (default 0)
 * - uncertainty_samples: number of posterior predictive samples (default 1000)
 */
public class ProphetConfig {

    public enum GrowthType { LINEAR, LOGISTIC }
    public enum SeasonalityMode { ADDITIVE, MULTIPLICATIVE }

    // Growth
    public GrowthType growth = GrowthType.LINEAR;
    public int nChangepoints = 25;
    public double changepointRange = 0.8;
    public double changepointPriorScale = 0.05;   // tau for Laplace prior on delta

    // Seasonality
    public SeasonalityMode seasonalityMode = SeasonalityMode.ADDITIVE;
    public double seasonalityPriorScale = 10.0;   // sigma_beta
    public int yearlyFourierOrder = 10;
    public int weeklyFourierOrder = 3;
    public boolean yearlySeasonality = true;
    public boolean weeklySeasonality = true;

    // Holidays
    public double holidaysPriorScale = 10.0;       // sigma_kappa

    // MCMC
    public int mcmcSamples = 0;                     // 0 = MAP only (Prophet default)
    public int mcmcWarmup = 500;
    public int mcmcChains = 4;                      // parallel MCMC chains (GPU parallel)
    public boolean useNUTS = true;                  // Use NUTS (Stan's sampler) vs MH

    // Uncertainty
    public int uncertaintySamples = 1000;           // posterior predictive samples
    public double sigmaObsPriorScale = 0.5;         // Half-Cauchy scale for sigma_obs

    // Optimization
    public int lbfgsMaxIter = 10000;
    public double lbfgsGradTol = 1e-8;
    public boolean verbose = true;

    // Logistic growth params
    public double cap = Double.MAX_VALUE;           // carrying capacity C
    public double floor = Double.NEGATIVE_INFINITY; // saturating minimum

    // Holiday definitions
    public List<HolidaySpec> holidays = new ArrayList<>();

    /**
     * Holiday specification — mirrors Prophet's holidays_table format.
     */
    public static class HolidaySpec {
        public final String name;
        public final double timestamp;   // day number
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
}
