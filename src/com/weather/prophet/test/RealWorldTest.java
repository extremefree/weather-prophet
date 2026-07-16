package com.weather.prophet.test;

import com.weather.prophet.core.*;
import com.weather.prophet.data.DataPoint;
import java.util.*;
import java.text.*;

/**
 * 16个真实场景全面测试
 * 每个场景模拟不同气候/数据条件，评估模型预测精度
 */
public class RealWorldTest {

    static final DecimalFormat DF2 = new DecimalFormat("0.00");
    static final DecimalFormat DF4 = new DecimalFormat("0.0000");
    static final DecimalFormat PCT = new DecimalFormat("0.0%");
    static final Random RNG = new Random(42);

    // ─── 结果记录 ───
    static class ScenarioResult {
        String name;
        int trainSize, testSize;
        double r2, mae, rmse, mape, noiseStd, maeOverSigma;
        double ci80Coverage;
        boolean passed;
        String note;
    }

    static List<ScenarioResult> allResults = new ArrayList<>();

    // ═══════════════════════════════════════════════════
    //  数据生成器
    // ═══════════════════════════════════════════════════

    /** 温带大陆性气候: 北京风格 — 四季分明, 冬冷夏热 */
    static List<DataPoint> genTemperateContinental(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = 13 + 20 * Math.cos(yearPhase - Math.PI) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 热带气候: 新加坡风格 — 全年炎热, 小幅季节波动 */
    static List<DataPoint> genTropical(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = 27 + 2 * Math.cos(yearPhase) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 极地气候: 冬季极寒 */
    static List<DataPoint> genArctic(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = -15 + 25 * Math.cos(yearPhase - Math.PI) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 地中海气候: 夏干冬湿 */
    static List<DataPoint> genMediterranean(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = 17 + 12 * Math.cos(yearPhase - Math.PI) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 季风气候: 明显干湿季 */
    static List<DataPoint> genMonsoon(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            // 温度有双峰结构（春夏和初秋各一个峰值）
            double y = 25 + 8 * Math.cos(yearPhase) + 3 * Math.cos(2 * yearPhase + 0.5)
                     + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 高原气候: 拉萨风格 — 日温差大 */
    static List<DataPoint> genPlateau(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = 5 + 14 * Math.cos(yearPhase - Math.PI) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 海洋性气候: 伦敦风格 — 全年温和 */
    static List<DataPoint> genOceanic(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = 10 + 8 * Math.cos(yearPhase - Math.PI) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 沙漠气候: 极端温差 */
    static List<DataPoint> genDesert(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = 30 + 20 * Math.cos(yearPhase - Math.PI) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 暖化趋势: 全球变暖模拟 */
    static List<DataPoint> genWarmingTrend(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double trend = 0.003 * t; // 每100天升温0.3°C
            double y = 13 + 18 * Math.cos(yearPhase - Math.PI) + trend + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 冷却趋势 */
    static List<DataPoint> genCoolingTrend(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double trend = -0.002 * t;
            double y = 20 + 12 * Math.cos(yearPhase - Math.PI) + trend + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 台风/飓风季节效应 */
    static List<DataPoint> genTyphoonEffect(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            // 夏秋有台风降温脉冲
            double typhoonPulse = 0;
            double dayOfYear = t % 365.25;
            if (dayOfYear > 180 && dayOfYear < 270 && RNG.nextDouble() < 0.05) {
                typhoonPulse = -8; // 台风降温
            }
            double y = 25 + 10 * Math.cos(yearPhase) + typhoonPulse + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 周周期+年周期 */
    static List<DataPoint> genWeeklyPlusYearly(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double weekPhase = 2 * Math.PI * (t % 7) / 7;
            // 工作日温度略高（城市热岛），周末略低
            double y = 15 + 15 * Math.cos(yearPhase - Math.PI) + 1.5 * Math.cos(weekPhase)
                     + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 乘法季节性: 振幅随温度升高而增大 */
    static List<DataPoint> genMultiplicativeSeasonal(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double baseTrend = 0.005 * t; // 上升趋势
            double base = 15 + baseTrend;
            // 季节振幅与 baseline 成比例
            double y = base * (1 + 0.3 * Math.cos(yearPhase - Math.PI)) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 阶跃变化: 城市热岛突然加剧 */
    static List<DataPoint> genStepChange(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        double stepSize = 3.0; // 突然升温3°C
        int stepTime = days / 2;
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double step = t >= stepTime ? stepSize : 0;
            double y = 13 + 18 * Math.cos(yearPhase - Math.PI) + step + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 非平稳振幅: 季节振幅随时间增大 */
    static List<DataPoint> genIncreasingAmplitude(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double ampGrowth = 1 + 0.5 * t / (double)days; // 振幅增长50%
            double y = 15 + 10 * ampGrowth * Math.cos(yearPhase - Math.PI) + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    /** 多重周期叠加: 年+半年+季 */
    static List<DataPoint> genMultiHarmonic(int days, double noiseStd) {
        List<DataPoint> data = new ArrayList<>();
        for (int t = 0; t < days; t++) {
            double yearPhase = 2 * Math.PI * (t % 365.25) / 365.25;
            double y = 15 + 12 * Math.cos(yearPhase - Math.PI)      // 年周期
                       + 4 * Math.cos(2 * yearPhase - 0.3)            // 半年周期
                       + 2 * Math.cos(4 * yearPhase + 0.7)            // 季周期
                       + RNG.nextGaussian() * noiseStd;
            data.add(new DataPoint(t, y));
        }
        return data;
    }

    // ═══════════════════════════════════════════════════
    //  评估函数
    // ═══════════════════════════════════════════════════

    static ScenarioResult runScenario(
            String name, List<DataPoint> allData, int testDays,
            ProphetConfig config, String note, double noiseStd) {

        ScenarioResult res = new ScenarioResult();
        res.name = name;
        res.note = note;

        int totalDays = allData.size();
        int trainDays = totalDays - testDays;
        res.trainSize = trainDays;
        res.testSize = testDays;

        // 分割训练/测试
        List<DataPoint> trainData = allData.subList(0, trainDays);
        List<DataPoint> testData = allData.subList(trainDays, totalDays);

        // 构建测试时间点
        double[] testT = new double[testDays];
        double[] testY = new double[testDays];
        for (int i = 0; i < testDays; i++) {
            testT[i] = testData.get(i).getTimestamp();
            testY[i] = testData.get(i).getValue();
        }

        // 训练模型
        ProphetModel model = new ProphetModel(config);
        long fitStart = System.currentTimeMillis();
        model.fit(trainData);
        long fitTime = System.currentTimeMillis() - fitStart;

        // 预测
        long predStart = System.currentTimeMillis();
        ProphetModel.ForecastResult forecast = model.predict(testT);
        long predTime = System.currentTimeMillis() - predStart;

        // 计算指标
        double ssRes = 0, ssTot = 0, sumAbsErr = 0, sumAbsPctErr = 0;
        int ciCount = 0;
        int validMape = 0;

        double yMean = 0;
        for (double v : testY) yMean += v;
        yMean /= testDays;

        for (int i = 0; i < testDays; i++) {
            double err = forecast.yhat[i] - testY[i];
            ssRes += err * err;
            ssTot += (testY[i] - yMean) * (testY[i] - yMean);
            sumAbsErr += Math.abs(err);
            if (Math.abs(testY[i]) > 1.0) {
                sumAbsPctErr += Math.abs(err / testY[i]);
                validMape++;
            }
            // CI coverage
            if (forecast.yhatLower != null && i < forecast.yhatLower.length) {
                if (testY[i] >= forecast.yhatLower[i] && testY[i] <= forecast.yhatUpper[i]) {
                    ciCount++;
                }
            }
        }

        res.r2 = 1 - ssRes / ssTot;
        res.mae = sumAbsErr / testDays;
        res.rmse = Math.sqrt(ssRes / testDays);
        res.mape = validMape > 0 ? sumAbsPctErr / validMape : Double.NaN;
        res.ci80Coverage = (double)ciCount / testDays;
        res.noiseStd = noiseStd;
        res.maeOverSigma = res.mae / noiseStd;

        // 通过标准: MAE/σ < 1.0 (模型误差低于噪声水平) AND CI coverage 50-98%
        // R²在短窗口+高噪声下不可靠: 如60天窗口内信号方差很小, R²会很低即使MAE/σ很好
        res.passed = res.maeOverSigma < 1.0
                  && res.ci80Coverage > 0.5 && res.ci80Coverage < 0.98;

        // 打印结果
        System.out.printf("  %-28s │ R²=%s  MAE/σ=%s  MAE=%-6s  CI80=%-5s  %s%s%n",
            name,
            DF4.format(res.r2),
            DF2.format(res.maeOverSigma),
            DF2.format(res.mae) + "°C",
            PCT.format(res.ci80Coverage),
            res.passed ? "PASS" : "FAIL",
            fitTime > 30000 ? " [slow:" + (fitTime/1000) + "s]" : ""
        );

        return res;
    }

    // ═══════════════════════════════════════════════════
    //  主程序
    // ═══════════════════════════════════════════════════

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║        Weather Prophet — 16 Real-World Scenario Comprehensive Test               ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        // ─── 默认配置 ───
        ProphetConfig baseLinear = new ProphetConfig();
        baseLinear.growth = ProphetConfig.GrowthType.LINEAR;
        baseLinear.nChangepoints = 25;
        baseLinear.changepointPriorScale = 0.05;
        baseLinear.yearlySeasonality = true;
        baseLinear.yearlyFourierOrder = 10;
        baseLinear.weeklySeasonality = false;
        baseLinear.mcmcSamples = 0; // L-BFGS only for speed
        baseLinear.uncertaintySamples = 500;

        ProphetConfig baseWithWeekly = new ProphetConfig();
        baseWithWeekly.growth = ProphetConfig.GrowthType.LINEAR;
        baseWithWeekly.nChangepoints = 25;
        baseWithWeekly.changepointPriorScale = 0.05;
        baseWithWeekly.yearlySeasonality = true;
        baseWithWeekly.yearlyFourierOrder = 10;
        baseWithWeekly.weeklySeasonality = true;
        baseWithWeekly.weeklyFourierOrder = 3;
        baseWithWeekly.mcmcSamples = 0;
        baseWithWeekly.uncertaintySamples = 500;

        ProphetConfig baseMultiplicative = new ProphetConfig();
        baseMultiplicative.growth = ProphetConfig.GrowthType.LINEAR;
        baseMultiplicative.nChangepoints = 25;
        baseMultiplicative.changepointPriorScale = 0.05;
        baseMultiplicative.yearlySeasonality = true;
        baseMultiplicative.yearlyFourierOrder = 10;
        baseMultiplicative.weeklySeasonality = false;
        baseMultiplicative.seasonalityMode = ProphetConfig.SeasonalityMode.MULTIPLICATIVE;
        baseMultiplicative.mcmcSamples = 0;
        baseMultiplicative.uncertaintySamples = 500;

        ProphetConfig baseLowTau = new ProphetConfig();
        baseLowTau.growth = ProphetConfig.GrowthType.LINEAR;
        baseLowTau.nChangepoints = 25;
        baseLowTau.changepointPriorScale = 0.001; // 强正则化, 抑制变点
        baseLowTau.yearlySeasonality = true;
        baseLowTau.yearlyFourierOrder = 10;
        baseLowTau.weeklySeasonality = false;
        baseLowTau.mcmcSamples = 0;
        baseLowTau.uncertaintySamples = 500;

        ProphetConfig baseFlat = new ProphetConfig();
        baseFlat.growth = ProphetConfig.GrowthType.FLAT;
        baseFlat.nChangepoints = 0;
        baseFlat.yearlySeasonality = true;
        baseFlat.yearlyFourierOrder = 10;
        baseFlat.weeklySeasonality = false;
        baseFlat.mcmcSamples = 0;
        baseFlat.uncertaintySamples = 500;

        ProphetConfig baseHighFourier = new ProphetConfig();
        baseHighFourier.growth = ProphetConfig.GrowthType.LINEAR;
        baseHighFourier.nChangepoints = 25;
        baseHighFourier.changepointPriorScale = 0.05;
        baseHighFourier.yearlySeasonality = true;
        baseHighFourier.yearlyFourierOrder = 20; // 高阶傅里叶
        baseHighFourier.weeklySeasonality = true;
        baseHighFourier.weeklyFourierOrder = 3;
        baseHighFourier.mcmcSamples = 0;
        baseHighFourier.uncertaintySamples = 500;

        int testDays = 60; // 每个场景预测60天

        // ══════════════════════════════════════════════════════════════════════
        //  场景1: 温带大陆性气候（北京风格）
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 1: Temperate Continental (Beijing) ──────────────────────────────────┐");
        allResults.add(runScenario(
            "Beijing 3yr σ=3", genTemperateContinental(1095, 3.0), testDays,
            baseLinear, "四季分明, 冬冷夏热", 3.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景2: 热带气候（新加坡风格）
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 2: Tropical (Singapore) ────────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Singapore 3yr σ=1", genTropical(1095, 1.0), testDays,
            baseLinear, "全年炎热, 小幅季节波动", 1.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景3: 极地气候
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 3: Arctic Climate ──────────────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Arctic 3yr σ=4", genArctic(1095, 4.0), testDays,
            baseLinear, "冬季极寒, 大幅季节波动", 4.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景4: 地中海气候
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 4: Mediterranean Climate ───────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Mediterranean 3yr σ=2", genMediterranean(1095, 2.0), testDays,
            baseLinear, "夏干冬湿, 温和季节变化", 2.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景5: 季风气候
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 5: Monsoon Climate ─────────────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Monsoon 3yr σ=2", genMonsoon(1095, 2.0), testDays,
            baseHighFourier, "双峰结构, 需高阶傅里叶", 2.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景6: 高原气候（拉萨风格）
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 6: Plateau Climate (Lhasa) ─────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Lhasa 3yr σ=4", genPlateau(1095, 4.0), testDays,
            baseLinear, "高原, 日温差大", 4.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景7: 海洋性气候（伦敦风格）
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 7: Oceanic Climate (London) ────────────────────────────────────────┐");
        allResults.add(runScenario(
            "London 3yr σ=2", genOceanic(1095, 2.0), testDays,
            baseLinear, "全年温和, 小季节振幅", 2.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景8: 沙漠气候
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 8: Desert Climate ──────────────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Desert 3yr σ=5", genDesert(1095, 5.0), testDays,
            baseLinear, "极端温差, 高噪声", 5.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景9: 全球暖化趋势
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 9: Global Warming Trend ────────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Warming 5yr σ=3", genWarmingTrend(1825, 3.0), testDays,
            baseLinear, "线性升温趋势 0.3°C/100天", 3.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景10: 冷却趋势
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 10: Cooling Trend ──────────────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Cooling 5yr σ=2", genCoolingTrend(1825, 2.0), testDays,
            baseLinear, "线性降温趋势", 2.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景11: 台风/飓风季节效应
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 11: Typhoon Season Effect ──────────────────────────────────────────┐");
        allResults.add(runScenario(
            "Typhoon 3yr σ=2", genTyphoonEffect(1095, 2.0), testDays,
            baseLinear, "夏秋台风脉冲降温", 2.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景12: 周+年周期
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 12: Weekly + Yearly Seasonality ───────────────────────────────────┐");
        allResults.add(runScenario(
            "Weekly+Yearly 2yr σ=2", genWeeklyPlusYearly(730, 2.0), testDays,
            baseWithWeekly, "工作日/周末温度差异 + 年周期", 2.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景13: 乘法季节性
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 13: Multiplicative Seasonality ────────────────────────────────────┐");
        allResults.add(runScenario(
            "Multiplicative 3yr σ=2", genMultiplicativeSeasonal(1095, 2.0), testDays,
            baseMultiplicative, "振幅随温度升高而增大", 2.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景14: 阶跃变化（城市热岛）
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 14: Step Change (Urban Heat Island) ───────────────────────────────┐");
        allResults.add(runScenario(
            "StepChange 4yr σ=3", genStepChange(1460, 3.0), testDays,
            baseLinear, "中期突然升温3°C", 3.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景15: 非平稳振幅 — 使用MIXED模式(加法+乘法)
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 15: Increasing Amplitude (MIXED mode) ─────────────────────────────┐");
        ProphetConfig mixedConfig = new ProphetConfig();
        mixedConfig.seasonalityMode = ProphetConfig.SeasonalityMode.MIXED;
        mixedConfig.yearlyFourierOrder = 10;
        mixedConfig.yearlySeasonalityPriorScale = 10.0;
        mixedConfig.verbose = false;
        allResults.add(runScenario(
            "AmpGrowth 5yr σ=3 MIXED", genIncreasingAmplitude(1825, 3.0), testDays,
            mixedConfig, "MIXED模式: 加法恒定振幅 + 乘法增长振幅", 3.0));

        // ══════════════════════════════════════════════════════════════════════
        //  场景16: 多重谐波叠加
        // ══════════════════════════════════════════════════════════════════════
        System.out.println("┌─── Scenario 16: Multi-Harmonic (Year+Half+Quarter) ────────────────────────────┐");
        allResults.add(runScenario(
            "MultiHarm 3yr σ=2", genMultiHarmonic(1095, 2.0), testDays,
            baseHighFourier, "年+半年+季度周期叠加", 2.0));

        // ══════════════════════════════════════════════════════════════════════
        //  汇总报告
        // ══════════════════════════════════════════════════════════════════════
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                          FINAL COMPREHENSIVE REPORT                                ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();

        int passCount = 0, failCount = 0;
        double sumR2 = 0, sumMaeOverSigma = 0, sumMae = 0, sumCi = 0;

        System.out.printf("%-4s %-28s │ %-6s %-8s %-8s %-8s %-6s %-6s%n",
            "#", "Scenario", "R²", "MAE/σ", "MAE(°C)", "CI80%", "Train", "Result");
        System.out.println("─────┼──────────────────────────────┼─────────────────────────────────────────────");

        for (int i = 0; i < allResults.size(); i++) {
            ScenarioResult r = allResults.get(i);
            System.out.printf("%-4d %-28s │ %-6s %-8s %-8s %-8s %-6d %-6s%n",
                i + 1,
                r.name,
                DF4.format(r.r2),
                DF2.format(r.maeOverSigma),
                DF2.format(r.mae),
                PCT.format(r.ci80Coverage),
                r.trainSize,
                r.passed ? "PASS" : "FAIL");

            sumR2 += r.r2;
            sumMaeOverSigma += r.maeOverSigma;
            sumMae += r.mae;
            sumCi += r.ci80Coverage;
            if (r.passed) passCount++; else failCount++;
        }

        int n = allResults.size();
        System.out.println("─────┼──────────────────────────────┼─────────────────────────────────────────────");
        System.out.printf("AVG  %-28s │ %-6s %-8s %-8s %-8s%n",
            "(" + n + " scenarios)",
            DF4.format(sumR2 / n),
            DF2.format(sumMaeOverSigma / n),
            DF2.format(sumMae / n),
            PCT.format(sumCi / n));

        System.out.println();
        System.out.println("═══════════════════════════════════════════════════════════════════════════");
        System.out.printf("  PASS: %d/%d  (%s)%n", passCount, n, PCT.format((double)passCount/n));
        System.out.printf("  FAIL: %d/%d  (%s)%n", failCount, n, PCT.format((double)failCount/n));
        System.out.println("═══════════════════════════════════════════════════════════════════════════");

        // 性能分级 (基于 MAE/σ)
        System.out.println();
        System.out.println("  ┌─── Performance Grading (MAE/σ) ────────────────┐");
        System.out.println("  │ Grade A (MAE/σ<0.5):  " + String.format("%-2d", allResults.stream().filter(r -> r.maeOverSigma < 0.5).count()) + " scenarios              │");
        System.out.println("  │ Grade B (MAE/σ<0.8):  " + String.format("%-2d", allResults.stream().filter(r -> r.maeOverSigma >= 0.5 && r.maeOverSigma < 0.8).count()) + " scenarios              │");
        System.out.println("  │ Grade C (MAE/σ<1.0):  " + String.format("%-2d", allResults.stream().filter(r -> r.maeOverSigma >= 0.8 && r.maeOverSigma < 1.0).count()) + " scenarios              │");
        System.out.println("  │ Grade D (MAE/σ≥1.0):  " + String.format("%-2d", allResults.stream().filter(r -> r.maeOverSigma >= 1.0).count()) + " scenarios              │");
        System.out.println("  └────────────────────────────────────────────────┘");

        // 失败场景详细分析
        if (failCount > 0) {
            System.out.println();
            System.out.println("  ┌─── Failed Scenarios Analysis ──────────────────┐");
            for (ScenarioResult r : allResults) {
                if (!r.passed) {
                    System.out.printf("  │ %-28s R²=%s MAE/σ=%s CI=%s%n",
                        r.name, DF4.format(r.r2), DF2.format(r.maeOverSigma), PCT.format(r.ci80Coverage));
                    System.out.printf("  │   → %s%n", r.note);
                    if (r.maeOverSigma >= 1.0) System.out.println("  │   → Root cause: MAE/σ ≥ 1.0 (error exceeds noise)");
                    if (r.ci80Coverage < 0.5) System.out.println("  │   → Root cause: CI80% < 50% (underfitting)");
                    if (r.ci80Coverage > 0.98) System.out.println("  │   → Root cause: CI80% > 98% (overly conservative CI)");
                }
            }
            System.out.println("  └────────────────────────────────────────────────┘");
        }

        // CI 校准评估
        System.out.println();
        System.out.println("  ┌─── CI Calibration Assessment ──────────────────┐");
        double avgCi = sumCi / n;
        String ciGrade = avgCi > 0.75 && avgCi < 0.85 ? "WELL CALIBRATED" :
                         avgCi > 0.85 ? "SLIGHTLY CONSERVATIVE" : "SLIGHTLY NARROW";
        System.out.printf("  │ Avg 80%% CI Coverage: %s → %s%n", PCT.format(avgCi), ciGrade);
        System.out.println("  └────────────────────────────────────────────────┘");

        System.out.println();
        System.out.println("  Test completed at: " + new Date());
    }
}
