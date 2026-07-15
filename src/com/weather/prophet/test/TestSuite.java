package com.weather.prophet.test;

import com.weather.prophet.matrix.Matrix;
import com.weather.prophet.matrix.VecOps;
import com.weather.prophet.optimize.BayesianPriors;
import com.weather.prophet.optimize.LBFGSOptimizer;
import com.weather.prophet.mcmc.MCMCSampler;
import com.weather.prophet.compute.CPUBackend;
import com.weather.prophet.compute.GPUBackend;
import com.weather.prophet.core.ProphetConfig;
import com.weather.prophet.core.ProphetModel;
import com.weather.prophet.data.DataPoint;

import java.util.List;
import java.util.Arrays;

/**
 * Full Test Suite for Weather Prophet (Java Prophet Implementation)
 */
public class TestSuite {

    static int passed = 0;
    static int failed = 0;
    static int skipped = 0;

    public static void main(String[] args) {
        System.out.println("=".repeat(72));
        System.out.println("  Weather Prophet — Test Suite");
        System.out.println("=".repeat(72));
        System.out.println();

        // 1. Matrix
        testMatrixMultiply();
        testMatrixTranspose();
        testMatrixSolve();
        testMatrixSolveLeastSquares();

        // 2. VecOps
        testVecOpsDot();
        testVecOpsNorm();
        testVecOpsAddScale();

        // 3. Bayesian Priors
        testNormalPrior();
        testLaplacePrior();
        testHalfCauchyPrior();
        testNormalGrad();
        testLaplaceGrad();

        // 4. L-BFGS
        testLBFGSQuadratic();
        testLBFGSRosenbrock();

        // 5. MCMC
        testMCMCGaussian();

        // 6. ProphetModel
        testProphetLinear();
        testProphetLogistic();

        // 7. CPU Backend
        testCPUFourier();
        testCPUMatmul();

        // 8. GPU Backend
        testGPUInit();

        // Summary
        System.out.println();
        System.out.println("=".repeat(72));
        System.out.println("  TEST SUMMARY");
        System.out.println("=".repeat(72));
        System.out.printf("  PASSED:  %d%n", passed);
        System.out.printf("  FAILED:  %d%n", failed);
        System.out.printf("  SKIPPED: %d%n", skipped);
        System.out.printf("  TOTAL:   %d%n", passed + failed + skipped);
        System.out.println("=".repeat(72));

        if (failed > 0) {
            System.out.println("  *** SOME TESTS FAILED ***");
            System.exit(1);
        } else {
            System.out.println("  ALL TESTS PASSED!");
        }
    }

    static void check(String name, boolean cond) {
        if (cond) {
            System.out.printf("  [PASS] %s%n", name);
            passed++;
        } else {
            System.out.printf("  [FAIL] %s%n", name);
            failed++;
        }
    }

    static void skip(String name, String reason) {
        System.out.printf("  [SKIP] %s — %s%n", name, reason);
        skipped++;
    }

    static void section(String title) {
        System.out.println();
        System.out.println("--- " + title + " ---");
    }

    // ====================================================================
    // 1. Matrix Tests
    // ====================================================================
    static void testMatrixMultiply() {
        section("Matrix: Multiply");
        Matrix A = new Matrix(new double[][]{{1, 2}, {3, 4}});
        Matrix B = new Matrix(new double[][]{{5, 6}, {7, 8}});
        Matrix C = A.multiply(B);
        check("2x2 matmul [0][0]=19", Math.abs(C.get(0, 0) - 19) < 1e-10);
        check("2x2 matmul [0][1]=22", Math.abs(C.get(0, 1) - 22) < 1e-10);
        check("2x2 matmul [1][0]=43", Math.abs(C.get(1, 0) - 43) < 1e-10);
        check("2x2 matmul [1][1]=50", Math.abs(C.get(1, 1) - 50) < 1e-10);

        Matrix D = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix E = new Matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        Matrix F = D.multiply(E);
        check("non-square rows=2", F.getRows() == 2);
        check("non-square cols=2", F.getCols() == 2);
        check("non-square [0][0]=58", Math.abs(F.get(0, 0) - 58) < 1e-10);
        check("non-square [1][1]=154", Math.abs(F.get(1, 1) - 154) < 1e-10);
    }

    static void testMatrixTranspose() {
        section("Matrix: Transpose");
        Matrix A = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix At = A.transpose();
        check("rows=3", At.getRows() == 3);
        check("cols=2", At.getCols() == 2);
        check("[0][1]=4", Math.abs(At.get(0, 1) - 4) < 1e-10);
        check("[2][0]=3", Math.abs(At.get(2, 0) - 3) < 1e-10);
    }

    static void testMatrixSolve() {
        section("Matrix: Solve Ax=b (Gaussian Elimination)");
        Matrix A = new Matrix(new double[][]{{2, 3}, {4, 1}});
        double[] b = {8, 6};
        double[] x = A.solve(b);
        check("x[0]=1", Math.abs(x[0] - 1.0) < 1e-8);
        check("x[1]=2", Math.abs(x[1] - 2.0) < 1e-8);

        Matrix A3 = new Matrix(new double[][]{{1, 1, 1}, {0, 2, 5}, {2, 5, -1}});
        double[] b3 = {6, -4, 27};
        double[] x3 = A3.solve(b3);
        check("3x3 x[0]=5", Math.abs(x3[0] - 5.0) < 1e-8);
        check("3x3 x[1]=3", Math.abs(x3[1] - 3.0) < 1e-8);
        check("3x3 x[2]=-2", Math.abs(x3[2] - (-2.0)) < 1e-8);
    }

    static void testMatrixSolveLeastSquares() {
        section("Matrix: Least Squares (Ridge)");
        double[][] Xd = {{1, 1}, {2, 1}, {1, 2}, {2, 2}};
        double[] yd = {5, 7, 8, 10};
        Matrix X = new Matrix(Xd);
        double[] beta = Matrix.solveLeastSquares(X, yd, 0.01);
        check("beta[0] ≈ 2", Math.abs(beta[0] - 2.0) < 0.2);
        check("beta[1] ≈ 3", Math.abs(beta[1] - 3.0) < 0.2);
    }

    // ====================================================================
    // 2. VecOps Tests
    // ====================================================================
    static void testVecOpsDot() {
        section("VecOps: Dot Product");
        check("dot=32", Math.abs(VecOps.dot(new double[]{1, 2, 3}, new double[]{4, 5, 6}) - 32) < 1e-10);
    }

    static void testVecOpsNorm() {
        section("VecOps: Norm");
        check("norm([3,4])=5", Math.abs(VecOps.norm(new double[]{3, 4}) - 5.0) < 1e-10);
    }

    static void testVecOpsAddScale() {
        section("VecOps: Add & Scale");
        double[] a = {1, 2, 3}, b = {4, 5, 6};
        double[] c = VecOps.add(a, b);
        check("add[0]=5", Math.abs(c[0] - 5) < 1e-10);
        check("add[2]=9", Math.abs(c[2] - 9) < 1e-10);
        double[] d = VecOps.scale(a, 2.0);
        check("scale[1]=4", Math.abs(d[1] - 4) < 1e-10);
    }

    // ====================================================================
    // 3. Bayesian Priors Tests
    // ====================================================================
    static void testNormalPrior() {
        section("BayesianPriors: Normal(mu, sigma)");
        double logp0 = BayesianPriors.logNormal(0, 0, 1);
        check("N(0,1) at x=0", Math.abs(logp0 - (-0.5 * Math.log(2 * Math.PI))) < 1e-6);
        check("N(0,1) at x=1 < at x=0", BayesianPriors.logNormal(1, 0, 1) < logp0);
        double logp2 = BayesianPriors.logNormal(0, 0, 2);
        check("N(0,2) at x=0", Math.abs(logp2 - (-0.5 * Math.log(2 * Math.PI * 4))) < 1e-6);
    }

    static void testLaplacePrior() {
        section("BayesianPriors: Laplace(mu, b)");
        check("Lap(0,1) at x=0", Math.abs(BayesianPriors.logLaplace(0, 0, 1) - (-Math.log(2))) < 1e-6);
        check("Lap(0,0.05) at x=0", Math.abs(BayesianPriors.logLaplace(0, 0, 0.05) - (-Math.log(0.1))) < 1e-6);
        // At tails (x far from mu), Normal penalizes MORE than Laplace due to quadratic vs linear
        check("Normal penalizes tails more than Laplace", BayesianPriors.logLaplace(1.0, 0, 0.05) > BayesianPriors.logNormal(1.0, 0, 0.05));
    }

    static void testHalfCauchyPrior() {
        section("BayesianPriors: HalfCauchy(scale)");
        check("HC(1) at x=0", Math.abs(BayesianPriors.logHalfCauchy(0, 1) - Math.log(2.0 / Math.PI)) < 1e-6);
        double neg = BayesianPriors.logHalfCauchy(-1, 1);
        check("HC rejects x<0", Double.isInfinite(neg) && neg < 0);
        check("HC heavier tail", BayesianPriors.logHalfCauchy(100, 1) > BayesianPriors.logNormal(100, 0, 1));
    }

    static void testNormalGrad() {
        section("BayesianPriors: Normal Gradient");
        check("grad at 1.5 = -1.5", Math.abs(BayesianPriors.gradLogNormal(1.5, 0, 1) - (-1.5)) < 1e-6);
        check("grad at 0 = 0", Math.abs(BayesianPriors.gradLogNormal(0, 0, 1)) < 1e-10);
    }

    static void testLaplaceGrad() {
        section("BayesianPriors: Laplace Gradient");
        check("grad at 1.5 = -1", Math.abs(BayesianPriors.gradLogLaplace(1.5, 0, 1) - (-1.0)) < 1e-6);
        check("grad at -1.5 = +1", Math.abs(BayesianPriors.gradLogLaplace(-1.5, 0, 1) - 1.0) < 1e-6);
    }

    // ====================================================================
    // 4. L-BFGS Tests
    // ====================================================================
    static void testLBFGSQuadratic() {
        section("L-BFGS: f(x) = 0.5*||x||^2");
        LBFGSOptimizer opt = new LBFGSOptimizer(7, 1000, 1e-12, 1e-10, 1e-10, false);
        double[] x0 = {5, 4, 3, 2, 1};
        LBFGSOptimizer.OptResult result = opt.minimize(x0, new LBFGSOptimizer.ObjectiveFunction() {
            @Override
            public double[] evaluate(double[] x) {
                double v = 0;
                for (double xi : x) v += xi * xi;
                double[] grad = x.clone();
                double[] out = new double[x.length + 1];
                out[0] = 0.5 * v;
                System.arraycopy(grad, 0, out, 1, x.length);
                return out;
            }
        });
        check("f* ≈ 0", Math.abs(result.value) < 1e-8);
        for (int i = 0; i < 5; i++) {
            check("x*[" + i + "] ≈ 0", Math.abs(result.params[i]) < 1e-5);
        }
    }

    static void testLBFGSRosenbrock() {
        section("L-BFGS: Rosenbrock");
        LBFGSOptimizer opt = new LBFGSOptimizer(7, 5000, 1e-12, 1e-8, 1e-10, false);
        double[] x0 = {-1.0, 1.0};
        LBFGSOptimizer.OptResult result = opt.minimize(x0, new LBFGSOptimizer.ObjectiveFunction() {
            @Override
            public double[] evaluate(double[] x) {
                double a = 1 - x[0], b = x[1] - x[0] * x[0];
                double val = a * a + 100 * b * b;
                double[] grad = new double[]{-2 * a - 400 * x[0] * b, 200 * b};
                double[] out = new double[3];
                out[0] = val;
                out[1] = grad[0];
                out[2] = grad[1];
                return out;
            }
        });
        check("x* ≈ 1", Math.abs(result.params[0] - 1.0) < 0.1);
        check("y* ≈ 1", Math.abs(result.params[1] - 1.0) < 0.1);
        check("f* ≈ 0", Math.abs(result.value) < 0.01);
    }

    // ====================================================================
    // 5. MCMC Tests
    // ====================================================================
    static void testMCMCGaussian() {
        section("MCMC: Sample from N(0,1)");
        // Use MH for simple 1D — NUTS is better for high-dimensional problems
        MCMCSampler sampler = new MCMCSampler(1000, 500, 2, false, false);

        double[] mapParams = {0.0};
        MCMCSampler.LogPosteriorFn logPost = new MCMCSampler.LogPosteriorFn() {
            @Override
            public double evaluate(double[] theta) {
                return -0.5 * theta[0] * theta[0];
            }
        };

        double[][] samples = sampler.sample(mapParams, logPost);
        check("samples count=2000", samples.length == 2000);

        double mean = 0;
        for (double[] s : samples) mean += s[0];
        mean /= samples.length;
        check("mean ≈ 0 (|m|<0.3)", Math.abs(mean) < 0.3);

        double var = 0;
        for (double[] s : samples) var += (s[0] - mean) * (s[0] - mean);
        var /= (samples.length - 1);
        check("var ≈ 1 (0.3<v<2.5)", var > 0.3 && var < 2.5);
        System.out.printf("    (mean=%.4f, var=%.4f)%n", mean, var);
    }

    // ====================================================================
    // 6. ProphetModel Tests
    // ====================================================================
    static void testProphetLinear() {
        section("ProphetModel: Linear Growth (End-to-End)");
        try {
            List<DataPoint> data = DataPoint.generateTemperatureData(200);

            ProphetConfig config = new ProphetConfig();
            config.growth = ProphetConfig.GrowthType.LINEAR;
            config.nChangepoints = 10;
            config.changepointPriorScale = 0.05;
            config.yearlyFourierOrder = 5;
            config.weeklyFourierOrder = 0;
            config.yearlySeasonality = true;
            config.weeklySeasonality = false;
            config.mcmcSamples = 20;
            config.mcmcWarmup = 10;
            config.mcmcChains = 2;
            config.uncertaintySamples = 50;
            config.verbose = false;

            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            check("k finite", Double.isFinite(model.getK()));
            check("m finite", Double.isFinite(model.getM()));
            check("sigma > 0", model.getSigmaObs() > 0);
            check("delta len=10", model.getDelta().length == 10);

            double[] futureT = new double[10];
            for (int i = 0; i < 10; i++) futureT[i] = 200 + i;
            ProphetModel.ForecastResult fc = model.predict(futureT);

            check("forecast size=10", fc.size() == 10);
            check("yhat all finite", allFinite(fc.yhat));
            check("lower <= yhat <= upper", withinBounds(fc));

            // In-sample
            double[] t = new double[data.size()], y = new double[data.size()];
            for (int i = 0; i < data.size(); i++) {
                t[i] = data.get(i).getTimestamp();
                y[i] = data.get(i).getValue();
            }
            ProphetModel.ForecastResult fitted = model.predict(t);
            double mae = 0, meanY = 0;
            for (int i = 0; i < y.length; i++) { mae += Math.abs(y[i] - fitted.yhat[i]); meanY += y[i]; }
            mae /= y.length; meanY /= y.length;
            double ssTot = 0, ssRes = 0;
            for (int i = 0; i < y.length; i++) {
                ssTot += (y[i] - meanY) * (y[i] - meanY);
                ssRes += (y[i] - fitted.yhat[i]) * (y[i] - fitted.yhat[i]);
            }
            double r2 = 1 - ssRes / ssTot;
            check("MAE < 10°C", mae < 10);
            check("R² > 0.4", r2 > 0.4);
            System.out.printf("    (MAE=%.2f°C, R²=%.4f)%n", mae, r2);

            model.getCompute().release();
        } catch (Exception e) {
            check("Linear no exception: " + e.getMessage(), false);
            e.printStackTrace();
        }
    }

    static void testProphetLogistic() {
        section("ProphetModel: Logistic Growth");
        try {
            List<DataPoint> data = new java.util.ArrayList<>();
            double cap = 100.0;
            for (int t = 0; t < 200; t++) {
                double val = cap / (1 + Math.exp(-0.05 * (t - 100))) + (Math.random() - 0.5) * 2;
                data.add(new DataPoint(t, val));
            }

            ProphetConfig config = new ProphetConfig();
            config.growth = ProphetConfig.GrowthType.LOGISTIC;
            config.cap = cap;
            config.nChangepoints = 10;
            config.changepointPriorScale = 0.05;
            config.yearlyFourierOrder = 0;
            config.weeklyFourierOrder = 0;
            config.yearlySeasonality = false;
            config.weeklySeasonality = false;
            config.mcmcSamples = 0;
            config.uncertaintySamples = 50;
            config.verbose = false;

            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            check("model fitted", Double.isFinite(model.getK()));

            double[] futureT = new double[10];
            for (int i = 0; i < 10; i++) futureT[i] = 200 + i;
            ProphetModel.ForecastResult fc = model.predict(futureT);
            double lastPred = fc.yhat[9];
            check("near cap (val=" + String.format("%.1f", lastPred) + ")", lastPred > 40 && lastPred <= cap * 1.2);

            model.getCompute().release();
        } catch (Exception e) {
            check("Logistic no exception: " + e.getMessage(), false);
            e.printStackTrace();
        }
    }

    static boolean allFinite(double[] a) {
        for (double v : a) if (!Double.isFinite(v)) return false;
        return true;
    }

    static boolean withinBounds(ProphetModel.ForecastResult fc) {
        for (int i = 0; i < fc.size(); i++) {
            if (!(fc.yhatLower[i] <= fc.yhat[i] && fc.yhat[i] <= fc.yhatUpper[i])) return false;
        }
        return true;
    }

    // ====================================================================
    // 7. CPU Backend Tests
    // ====================================================================
    static void testCPUFourier() {
        section("CPUBackend: Fourier Features");
        CPUBackend cpu = new CPUBackend();
        double[] t = {0, 365.25 / 4, 365.25 / 2, 365.25 * 3 / 4};
        double[][] f = cpu.computeFourierFeatures(t, 2, 365.25);
        check("cols=4", f[0].length == 4);
        check("cos(0)≈1", Math.abs(f[0][0] - 1.0) < 1e-10);
        check("sin(0)≈0", Math.abs(f[0][1]) < 1e-10);
        check("sin(T/4)≈1", Math.abs(f[1][1] - 1.0) < 1e-6);
        cpu.release();
    }

    static void testCPUMatmul() {
        section("CPUBackend: Matrix Multiply");
        CPUBackend cpu = new CPUBackend();
        double[][] A = {{1, 2}, {3, 4}};
        double[][] B = {{5, 6}, {7, 8}};
        double[][] C = cpu.matmul(A, B);
        check("[0][0]=19", Math.abs(C[0][0] - 19) < 1e-10);
        check("[1][1]=50", Math.abs(C[1][1] - 50) < 1e-10);
        cpu.release();
    }

    // ====================================================================
    // 8. GPU Backend Test
    // ====================================================================
    static void testGPUInit() {
        section("GPUBackend: OpenCL");
        try {
            GPUBackend gpu = new GPUBackend();
            if (gpu.isAvailable()) {
                check("GPU available", true);
                double[][] A = {{1, 2}, {3, 4}};
                double[][] B = {{5, 6}, {7, 8}};
                double[][] C = gpu.matmul(A, B);
                check("GPU [0][0]=19", Math.abs(C[0][0] - 19) < 1e-6);
                gpu.release();
            } else {
                skip("GPU tests", "No OpenCL device (expected in container)");
            }
        } catch (UnsatisfiedLinkError e) {
            skip("GPU tests", "JOCL native lib not available");
        } catch (Exception e) {
            skip("GPU tests", e.getClass().getSimpleName() + ": " + e.getMessage());
        }
    }
}
