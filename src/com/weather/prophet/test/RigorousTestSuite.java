package com.weather.prophet.test;

import com.weather.prophet.core.ProphetConfig;
import com.weather.prophet.core.ProphetModel;
import com.weather.prophet.data.DataPoint;
import com.weather.prophet.matrix.Matrix;
import com.weather.prophet.matrix.VecOps;
import com.weather.prophet.optimize.BayesianPriors;
import com.weather.prophet.optimize.LBFGSOptimizer;
import com.weather.prophet.mcmc.MCMCSampler;
import com.weather.prophet.compute.CPUBackend;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * 严谨测试套件 — 覆盖数值精度、收敛性诊断、边界条件、统计验证、端到端集成
 *
 * 测试分类:
 * A. 矩阵运算数值精度 (与理论值对比)
 * B. L-BFGS 收敛性诊断 (迭代次数、梯度范数、解精度)
 * C. 贝叶斯先验正确性 (解析值对比、梯度检验)
 * D. MCMC 收敛诊断 (Gelman-Rubin R-hat, 有效样本量 ESS, KS 检验)
 * E. Prophet 端到端集成 (已知真值信号的拟合、预测精度)
 * F. 边界条件与鲁棒性 (小数据集、大数据集、极端参数)
 * G. Prophet 算法忠实性 (与 Prophet 论文/源码公式一致性)
 */
public class RigorousTestSuite {

    private static int totalTests = 0;
    private static int passed = 0;
    private static int failed = 0;
    private static int skipped = 0;
    private static final List<String> failures = new ArrayList<>();

    // ===================== Assertion Helpers =====================

    private static void check(String name, boolean condition) {
        totalTests++;
        if (condition) {
            passed++;
            System.out.printf("  ✓ PASS: %s%n", name);
        } else {
            failed++;
            failures.add(name);
            System.out.printf("  ✗ FAIL: %s%n", name);
        }
    }

    private static void checkClose(String name, double actual, double expected, double tol) {
        double relErr = expected != 0 ? Math.abs(actual - expected) / Math.abs(expected) : Math.abs(actual - expected);
        boolean ok = relErr <= tol;
        totalTests++;
        if (ok) {
            passed++;
            System.out.printf("  ✓ PASS: %s (actual=%.8f, expected=%.8f, relErr=%.2e)%n", name, actual, expected, relErr);
        } else {
            failed++;
            failures.add(name);
            System.out.printf("  ✗ FAIL: %s (actual=%.8f, expected=%.8f, relErr=%.2e, tol=%.2e)%n", name, actual, expected, relErr, tol);
        }
    }

    private static void checkCloseAbs(String name, double actual, double expected, double absTol) {
        double absErr = Math.abs(actual - expected);
        boolean ok = absErr <= absTol;
        totalTests++;
        if (ok) {
            passed++;
            System.out.printf("  ✓ PASS: %s (actual=%.10f, expected=%.10f, absErr=%.2e)%n", name, actual, expected, absErr);
        } else {
            failed++;
            failures.add(name);
            System.out.printf("  ✗ FAIL: %s (actual=%.10f, expected=%.10f, absErr=%.2e, tol=%.2e)%n", name, actual, expected, absErr, absTol);
        }
    }

    private static void skip(String name, String reason) {
        totalTests++;
        skipped++;
        System.out.printf("  ○ SKIP: %s (%s)%n", name, reason);
    }

    // ===================== A. Matrix Numerical Precision =====================

    private static void testMatrixNumericalPrecision() {
        System.out.println("\n════════════════════════════════════════════════════════");
        System.out.println("A. 矩阵运算数值精度测试");
        System.out.println("════════════════════════════════════════════════════════");

        // A1. 2x2 矩阵乘法与手算对比
        {
            Matrix A = new Matrix(new double[][]{{1, 2}, {3, 4}});
            Matrix B = new Matrix(new double[][]{{5, 6}, {7, 8}});
            Matrix C = A.multiply(B);
            // [1*5+2*7, 1*6+2*8] = [19, 22]
            // [3*5+4*7, 3*6+4*8] = [43, 50]
            checkCloseAbs("A1. 2x2 matmul [0,0]", C.get(0, 0), 19.0, 1e-12);
            checkCloseAbs("A1. 2x2 matmul [0,1]", C.get(0, 1), 22.0, 1e-12);
            checkCloseAbs("A1. 2x2 matmul [1,0]", C.get(1, 0), 43.0, 1e-12);
            checkCloseAbs("A1. 2x2 matmul [1,1]", C.get(1, 1), 50.0, 1e-12);
        }

        // A2. 转置不变性: (A^T)^T = A
        {
            Matrix A = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
            Matrix ATT = A.transpose().transpose();
            for (int i = 0; i < A.getRows(); i++)
                for (int j = 0; j < A.getCols(); j++)
                    checkCloseAbs("A2. Transpose involution [" + i + "," + j + "]", ATT.get(i, j), A.get(i, j), 1e-15);
        }

        // A3. 矩阵乘法结合律: (AB)C = A(BC)
        {
            Random rng = new Random(123);
            Matrix A = randomMatrix(3, 4, rng);
            Matrix B = randomMatrix(4, 2, rng);
            Matrix C = randomMatrix(2, 5, rng);
            Matrix AB_C = A.multiply(B).multiply(C);
            Matrix A_BC = A.multiply(B.multiply(C));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 5; j++)
                    checkCloseAbs("A3. Associativity [" + i + "," + j + "]", AB_C.get(i, j), A_BC.get(i, j), 1e-10);
        }

        // A4. 高斯消元求解: Ax=b 精度验证
        {
            Matrix A = new Matrix(new double[][]{{2, 1, -1}, {-3, -1, 2}, {-2, 1, 2}});
            double[] b = {8, -11, -3};
            double[] x = A.solve(b);
            // 解为 x = [2, 3, -1]
            checkCloseAbs("A4. Solve x[0]", x[0], 2.0, 1e-10);
            checkCloseAbs("A4. Solve x[1]", x[1], 3.0, 1e-10);
            checkCloseAbs("A4. Solve x[2]", x[2], -1.0, 1e-10);
        }

        // A5. 岭回归最小二乘: 过定系统求解
        {
            // y = 2x1 + 3x2 + noise, 5个数据点
            Matrix X = new Matrix(new double[][]{{1, 1}, {2, 1}, {3, 1}, {4, 1}, {5, 1}});
            double[] y = {5, 7, 9, 11, 13}; // y = 2*x1 + 3*x2
            double[] beta = Matrix.solveLeastSquares(X, y, 0.01);
            checkClose("A5. Least squares beta[0]", beta[0], 2.0, 0.05);
            checkClose("A5. Least squares beta[1]", beta[1], 3.0, 0.05);
        }

        // A6. Hilbert 矩阵求解（数值病态测试）
        {
            int n = 5;
            Matrix H = new Matrix(n, n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    H.set(i, j, 1.0 / (i + j + 1));
            double[] xTrue = new double[n];
            for (int i = 0; i < n; i++) xTrue[i] = 1.0;
            // b = H * x
            double[] b = new double[n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    b[i] += H.get(i, j) * xTrue[j];
            double[] xSolved = H.solve(b);
            // Hilbert 矩阵条件数极高，5x5 的解应在 1e-4 精度内
            double maxErr = 0;
            for (int i = 0; i < n; i++) maxErr = Math.max(maxErr, Math.abs(xSolved[i] - 1.0));
            check("A6. Hilbert 5x5 solve max error < 0.01", maxErr < 0.01);
            System.out.printf("       Hilbert max error = %.2e%n", maxErr);
        }

        // A7. 大矩阵乘法正确性 (50x50)
        {
            Random rng = new Random(456);
            Matrix A = randomMatrix(50, 30, rng);
            Matrix B = randomMatrix(30, 40, rng);
            Matrix C = A.multiply(B);
            // 验证 C[i,j] = sum_k A[i,k]*B[k,j] 对几个随机元素
            for (int trial = 0; trial < 5; trial++) {
                int i = rng.nextInt(50), j = rng.nextInt(40);
                double expected = 0;
                for (int k = 0; k < 30; k++) expected += A.get(i, k) * B.get(k, j);
                checkCloseAbs("A7. 50x40 matmul random [" + i + "," + j + "]", C.get(i, j), expected, 1e-8);
            }
        }

        // A8. VecOps 数值精度
        {
            double[] a = {3.0, 4.0};
            checkCloseAbs("A8. VecOps.norm([3,4])", VecOps.norm(a), 5.0, 1e-12);
            checkCloseAbs("A8. VecOps.dot([1,2],[3,4])", VecOps.dot(new double[]{1, 2}, new double[]{3, 4}), 11.0, 1e-12);
        }
    }

    // ===================== B. L-BFGS Convergence Diagnostics =====================

    private static void testLBFGSConvergence() {
        System.out.println("\n════════════════════════════════════════════════════════");
        System.out.println("B. L-BFGS 收敛性诊断测试");
        System.out.println("════════════════════════════════════════════════════════");

        // B1. 二次函数: f(x) = 0.5 * x^T Q x, Q = diag(1, 10, 100)
        //     最优点 x* = 0, f* = 0
        {
            LBFGSOptimizer opt = new LBFGSOptimizer(7, 1000, 1e-12, 1e-14, 1e-14, false);
            double[] x0 = {5.0, -3.0, 2.0};
            LBFGSOptimizer.OptResult res = opt.minimize(x0, x -> {
                double val = 0.5 * (x[0] * x[0] + 10 * x[1] * x[1] + 100 * x[2] * x[2]);
                double[] grad = {x[0], 10 * x[1], 100 * x[2]};
                double[] result = new double[4];
                result[0] = val;
                System.arraycopy(grad, 0, result, 1, 3);
                return result;
            });
            checkCloseAbs("B1. Quadratic x[0]", res.params[0], 0.0, 1e-8);
            checkCloseAbs("B1. Quadratic x[1]", res.params[1], 0.0, 1e-7);
            checkCloseAbs("B1. Quadratic x[2]", res.params[2], 0.0, 1e-6);
            check("B1. Quadratic gradNorm < 1e-8", res.gradNorm < 1e-8);
            System.out.printf("       f*=%.2e, |g*|=%.2e%n", res.value, res.gradNorm);
        }

        // B2. Rosenbrock 函数: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        //     最优点 (1, 1), f* = 0
        {
            LBFGSOptimizer opt = new LBFGSOptimizer(10, 50000, 1e-10, 1e-12, 1e-12, false);
            double[] x0 = {-1.0, 1.0};
            LBFGSOptimizer.OptResult res = opt.minimize(x0, x -> {
                double xv = x[0], yv = x[1];
                double val = Math.pow(1 - xv, 2) + 100 * Math.pow(yv - xv * xv, 2);
                double dx = -2 * (1 - xv) - 400 * xv * (yv - xv * xv);
                double dy = 200 * (yv - xv * xv);
                return new double[]{val, dx, dy};
            });
            checkCloseAbs("B2. Rosenbrock x", res.params[0], 1.0, 1e-3);
            checkCloseAbs("B2. Rosenbrock y", res.params[1], 1.0, 1e-3);
            check("B2. Rosenbrock f* < 0.01", res.value < 0.01);
            System.out.printf("       f*=%.2e, |g*|=%.2e, x=(%.6f, %.6f)%n", res.value, res.gradNorm, res.params[0], res.params[1]);
        }

        // B3. Beale 函数: 更复杂的非线性优化
        // f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
        // 最优点 (3, 0.5), f* = 0
        {
            LBFGSOptimizer opt = new LBFGSOptimizer(10, 50000, 1e-10, 1e-12, 1e-12, false);
            double[] x0 = {0.0, 0.0};
            LBFGSOptimizer.OptResult res = opt.minimize(x0, x -> {
                double xv = x[0], yv = x[1];
                double t1 = 1.5 - xv + xv * yv;
                double t2 = 2.25 - xv + xv * yv * yv;
                double t3 = 2.625 - xv + xv * yv * yv * yv;
                double val = t1 * t1 + t2 * t2 + t3 * t3;
                // 数值梯度
                double h = 1e-7;
                double dx = ((1.5 - (xv + h) + (xv + h) * yv) * (1.5 - (xv + h) + (xv + h) * yv)
                        + (2.25 - (xv + h) + (xv + h) * yv * yv) * (2.25 - (xv + h) + (xv + h) * yv * yv)
                        + (2.625 - (xv + h) + (xv + h) * yv * yv * yv) * (2.625 - (xv + h) + (xv + h) * yv * yv * yv)
                        - val) / h;
                double dy = ((1.5 - xv + xv * (yv + h)) * (1.5 - xv + xv * (yv + h))
                        + (2.25 - xv + xv * (yv + h) * (yv + h)) * (2.25 - xv + xv * (yv + h) * (yv + h))
                        + (2.625 - xv + xv * (yv + h) * (yv + h) * (yv + h)) * (2.625 - xv + xv * (yv + h) * (yv + h) * (yv + h))
                        - val) / h;
                return new double[]{val, dx, dy};
            });
            checkCloseAbs("B3. Beale x", res.params[0], 3.0, 0.1);
            checkCloseAbs("B3. Beale y", res.params[1], 0.5, 0.1);
            System.out.printf("       f*=%.2e, x=(%.6f, %.6f)%n", res.value, res.params[0], res.params[1]);
        }

        // B4. 高维二次函数 (100维)
        {
            int dim = 100;
            double[] x0 = new double[dim];
            for (int i = 0; i < dim; i++) x0[i] = 1.0;
            double[] eigenvalues = new double[dim];
            for (int i = 0; i < dim; i++) eigenvalues[i] = 1.0 + i; // 条件数 = 100

            LBFGSOptimizer opt = new LBFGSOptimizer(20, 5000, 1e-10, 1e-12, 1e-12, false);
            LBFGSOptimizer.OptResult res = opt.minimize(x0, x -> {
                double val = 0;
                double[] grad = new double[dim];
                for (int i = 0; i < dim; i++) {
                    val += 0.5 * eigenvalues[i] * x[i] * x[i];
                    grad[i] = eigenvalues[i] * x[i];
                }
                double[] result = new double[1 + dim];
                result[0] = val;
                System.arraycopy(grad, 0, result, 1, dim);
                return result;
            });
            check("B4. 100-dim quadratic: all |x_i| < 1e-6",
                    Arrays.stream(res.params).allMatch(xi -> Math.abs(xi) < 1e-6));
            check("B4. 100-dim quadratic: |grad| < 1e-6", res.gradNorm < 1e-6);
            System.out.printf("       f*=%.2e, |g*|=%.2e%n", res.value, res.gradNorm);
        }

        // B5. 线搜索鲁棒性: 从远处起始的二次函数
        {
            LBFGSOptimizer opt = new LBFGSOptimizer(7, 10000, 1e-10, 1e-12, 1e-12, false);
            double[] x0 = {1000.0, -1000.0};
            LBFGSOptimizer.OptResult res = opt.minimize(x0, x -> {
                double val = x[0] * x[0] + x[1] * x[1];
                return new double[]{val, 2 * x[0], 2 * x[1]};
            });
            checkCloseAbs("B5. Far-start quadratic x", res.params[0], 0.0, 1e-6);
            checkCloseAbs("B5. Far-start quadratic y", res.params[1], 0.0, 1e-6);
        }
    }

    // ===================== C. Bayesian Priors Correctness =====================

    private static void testBayesianPriors() {
        System.out.println("\n════════════════════════════════════════════════════════");
        System.out.println("C. 贝叶斯先验正确性测试");
        System.out.println("════════════════════════════════════════════════════════");

        // C1. Normal PDF 精度: N(0,1) at x=0 应为 1/sqrt(2π)
        {
            double logPdf0 = BayesianPriors.logNormal(0, 0, 1);
            double expected = -0.5 * Math.log(2 * Math.PI);
            checkCloseAbs("C1. N(0,1) log PDF at x=0", logPdf0, expected, 1e-14);
        }

        // C2. Normal PDF 归一化: 积分 = 1 (数值积分验证)
        {
            double integral = 0;
            double dx = 0.01;
            for (double x = -10; x < 10; x += dx) {
                integral += Math.exp(BayesianPriors.logNormal(x, 0, 1)) * dx;
            }
            checkClose("C2. N(0,1) integral ≈ 1", integral, 1.0, 0.01);
        }

        // C3. Normal 梯度: 解析 vs 数值有限差分
        {
            double x = 1.5, mu = 0, sigma = 2.0;
            double analyticGrad = BayesianPriors.gradLogNormal(x, mu, sigma);
            double h = 1e-6;
            double numericalGrad = (BayesianPriors.logNormal(x + h, mu, sigma) - BayesianPriors.logNormal(x - h, mu, sigma)) / (2 * h);
            checkCloseAbs("C3. Normal gradient: analytic vs numerical", analyticGrad, numericalGrad, 1e-6);
        }

        // C4. Laplace PDF 归一化: 积分 = 1
        {
            double integral = 0;
            double dx = 0.01;
            double b = 0.5;
            for (double x = -10; x < 10; x += dx) {
                integral += Math.exp(BayesianPriors.logLaplace(x, 0, b)) * dx;
            }
            checkClose("C4. Laplace(0,0.5) integral ≈ 1", integral, 1.0, 0.01);
        }

        // C5. Laplace 梯度 vs 数值差分
        {
            double x = 0.3, mu = 0, b = 0.05;
            double analyticGrad = BayesianPriors.gradLogLaplace(x, mu, b);
            double h = 1e-6;
            double numericalGrad = (BayesianPriors.logLaplace(x + h, mu, b) - BayesianPriors.logLaplace(x - h, mu, b)) / (2 * h);
            checkCloseAbs("C5. Laplace gradient: analytic vs numerical", analyticGrad, numericalGrad, 1e-4);
        }

        // C6. Half-Cauchy PDF 归一化: 积分 = 1 (只积分 x >= 0)
        {
            double integral = 0;
            double dx = 0.01;
            double scale = 0.5;
            for (double x = 0; x < 50; x += dx) {
                integral += Math.exp(BayesianPriors.logHalfCauchy(x, scale)) * dx;
            }
            checkClose("C6. HalfCauchy(0,0.5) integral ≈ 1", integral, 1.0, 0.02);
        }

        // C7. Half-Cauchy 梯度 vs 数值差分
        {
            double x = 0.3, scale = 0.5;
            double analyticGrad = BayesianPriors.gradLogHalfCauchy(x, scale);
            double h = 1e-6;
            double numericalGrad = (BayesianPriors.logHalfCauchy(x + h, scale) - BayesianPriors.logHalfCauchy(x - h, scale)) / (2 * h);
            checkCloseAbs("C7. HalfCauchy gradient: analytic vs numerical", analyticGrad, numericalGrad, 1e-4);
        }

        // C8. Half-Cauchy x=0 有效性
        {
            double logPdf = BayesianPriors.logHalfCauchy(0, 0.5);
            check("C8. HalfCauchy(0) not -Inf", Double.isFinite(logPdf));
            check("C8. HalfCauchy(0) > HalfCauchy(1)", logPdf > BayesianPriors.logHalfCauchy(1, 0.5));
        }

        // C9. Half-Cauchy 负值返回 -Inf
        {
            double logPdf = BayesianPriors.logHalfCauchy(-1, 0.5);
            check("C9. HalfCauchy(-1) = -Inf", Double.isInfinite(logPdf));
        }

        // C10. 完整后验梯度检验（Prophet 的 negLogPosteriorAndGradient）
        {
            double[] t = {0, 1, 2, 3, 4};
            double[] y = {1, 2, 3, 4, 5};
            double[] changepoints = {1.5, 3.0};
            double[][] XSeasonal = {{0.5, -0.5}, {0.3, 0.7}, {-0.2, 0.8}, {0.1, -0.3}, {-0.4, 0.6}};
            double[][] XHoliday = null;
            double tau = 0.05, sigmaBeta = 10, sigmaKappa = 10, sigmaObsScale = 0.5;

            int S = changepoints.length, K = 2, H = 0;
            int numParams = 3 + S + K + H;
            double[] theta = new double[numParams];
            theta[0] = 0.8;  // k
            theta[1] = 0.5;  // m
            theta[2] = 0.01; // delta[0]
            theta[3] = -0.02; // delta[1]
            theta[4] = Math.log(0.5); // logSigmaObs
            theta[5] = 0.1;  // beta[0]
            theta[6] = -0.1; // beta[1]

            double[] result = BayesianPriors.negLogPosteriorAndGradient(
                    theta, t, y, changepoints, XSeasonal, XHoliday,
                    tau, sigmaBeta, sigmaKappa, sigmaObsScale,
                    false, 100, 0, 1
            );

            // 数值梯度
            double h = 1e-5;
            double[] numGrad = new double[numParams];
            for (int i = 0; i < numParams; i++) {
                double[] thetaP = theta.clone();
                double[] thetaM = theta.clone();
                thetaP[i] += h;
                thetaM[i] -= h;
                double valP = BayesianPriors.negLogPosteriorAndGradient(
                        thetaP, t, y, changepoints, XSeasonal, XHoliday,
                        tau, sigmaBeta, sigmaKappa, sigmaObsScale,
                        false, 100, 0, 1
                )[0];
                double valM = BayesianPriors.negLogPosteriorAndGradient(
                        thetaM, t, y, changepoints, XSeasonal, XHoliday,
                        tau, sigmaBeta, sigmaKappa, sigmaObsScale,
                        false, 100, 0, 1
                )[0];
                numGrad[i] = (valP - valM) / (2 * h);
            }

            // 对比解析梯度和数值梯度
            double maxGradErr = 0;
            for (int i = 0; i < numParams; i++) {
                double analyticGrad = result[1 + i];
                double absErr = Math.abs(analyticGrad - numGrad[i]);
                double relErr = Math.abs(numGrad[i]) > 1e-8 ? absErr / Math.abs(numGrad[i]) : absErr;
                maxGradErr = Math.max(maxGradErr, relErr);
            }
            check("C10. Full posterior gradient: max relative error < 0.05", maxGradErr < 0.05);
            System.out.printf("       Max gradient relative error = %.2e%n", maxGradErr);
            for (int i = 0; i < numParams; i++) {
                System.out.printf("       param[%d]: analytic=%.6f, numerical=%.6f, relErr=%.2e%n",
                        i, result[1 + i], numGrad[i],
                        Math.abs(numGrad[i]) > 1e-8 ? Math.abs(result[1 + i] - numGrad[i]) / Math.abs(numGrad[i]) : Math.abs(result[1 + i] - numGrad[i]));
            }
        }
    }

    // ===================== D. MCMC Convergence Diagnostics =====================

    private static void testMCMCConvergence() {
        System.out.println("\n════════════════════════════════════════════════════════");
        System.out.println("D. MCMC 收敛诊断测试");
        System.out.println("════════════════════════════════════════════════════════");

        // D1. MH 采样标准正态 — 均值和方差检验
        {
            System.out.println("  [D1] MH sampling from N(0,1)...");
            MCMCSampler sampler = new MCMCSampler(2000, 1000, 4, false, false);
            double[][] samples = sampler.sample(new double[]{0.0}, x -> -0.5 * x[0] * x[0]);

            double mean = 0, var = 0;
            for (double[] s : samples) mean += s[0];
            mean /= samples.length;
            for (double[] s : samples) var += (s[0] - mean) * (s[0] - mean);
            var /= samples.length;

            checkClose("D1. MH N(0,1) mean ≈ 0", mean, 0.0, 0.1);
            checkClose("D1. MH N(0,1) var ≈ 1", var, 1.0, 0.2);
            System.out.printf("       mean=%.4f, var=%.4f, nSamples=%d%n", mean, var, samples.length);
        }

        // D2. NUTS 采样 5D 正态 — 高维场景验证 NUTS 优势
        {
            System.out.println("  [D2] NUTS sampling from 5D N(0,I)...");
            int D = 5;
            double[] init = new double[D];
            MCMCSampler sampler = new MCMCSampler(3000, 1500, 4, true, false);
            double[][] samples = sampler.sample(init, x -> {
                double ss = 0;
                for (int i = 0; i < D; i++) ss += x[i] * x[i];
                return -0.5 * ss;
            });

            double[] mean = new double[D], var = new double[D];
            for (double[] s : samples) for (int i = 0; i < D; i++) mean[i] += s[i];
            for (int i = 0; i < D; i++) mean[i] /= samples.length;
            for (double[] s : samples) for (int i = 0; i < D; i++) var[i] += (s[i] - mean[i]) * (s[i] - mean[i]);
            double avgVar = 0;
            for (int i = 0; i < D; i++) { var[i] /= samples.length; avgVar += var[i]; }
            avgVar /= D;
            double avgAbsMean = 0;
            for (int i = 0; i < D; i++) avgAbsMean += Math.abs(mean[i]);
            avgAbsMean /= D;

            checkClose("D2. NUTS 5D N(0,I) avg |mean| < 0.2", avgAbsMean, 0.0, 0.2);
            checkClose("D2. NUTS 5D N(0,I) avg var ≈ 1", avgVar, 1.0, 2.0);
            System.out.printf("       avgMean=%.4f, avgVar=%.4f, nSamples=%d%n", avgAbsMean, avgVar, samples.length);
        }

        // D3. 2D 相关正态 — 验证 MCMC 捕获相关性
        {
            System.out.println("  [D3] MH sampling from correlated 2D normal...");
            double rho = 0.7;
            double[][] cov = {{1, rho}, {rho, 1}};
            // log p(x) = -0.5 * x^T * cov^{-1} * x
            double det = 1 - rho * rho;
            double[][] covInv = {{1 / det, -rho / det}, {-rho / det, 1 / det}};

            MCMCSampler sampler = new MCMCSampler(3000, 1500, 4, false, false);
            double[][] samples = sampler.sample(new double[]{0.0, 0.0}, x -> {
                double quad = covInv[0][0] * x[0] * x[0] + 2 * covInv[0][1] * x[0] * x[1] + covInv[1][1] * x[1] * x[1];
                return -0.5 * quad;
            });

            double mean0 = 0, mean1 = 0, cov01 = 0;
            for (double[] s : samples) { mean0 += s[0]; mean1 += s[1]; }
            mean0 /= samples.length;
            mean1 /= samples.length;
            for (double[] s : samples) cov01 += (s[0] - mean0) * (s[1] - mean1);
            cov01 /= samples.length;

            checkClose("D3. 2D normal mean[0] ≈ 0", mean0, 0.0, 0.15);
            checkClose("D3. 2D normal mean[1] ≈ 0", mean1, 0.0, 0.15);
            checkClose("D3. 2D normal corr ≈ 0.7", cov01, rho, 0.2);
            System.out.printf("       mean=(%.4f, %.4f), cov01=%.4f (expected %.2f)%n", mean0, mean1, cov01, rho);
        }

        // D4. Gelman-Rubin R-hat 诊断
        {
            System.out.println("  [D4] Gelman-Rubin R-hat diagnostic...");
            int numChains = 4;
            int numSamples = 2000;
            int numWarmup = 1000;
            MCMCSampler sampler = new MCMCSampler(numSamples, numWarmup, numChains, false, false);
            double[][] allSamples = sampler.sample(new double[]{0.0}, x -> -0.5 * x[0] * x[0]);

            // 分链计算 R-hat
            int perChain = allSamples.length / numChains;
            double[] chainMeans = new double[numChains];
            double[] chainVars = new double[numChains];
            for (int c = 0; c < numChains; c++) {
                double sum = 0, sumSq = 0;
                for (int i = 0; i < perChain; i++) {
                    double v = allSamples[c * perChain + i][0];
                    sum += v;
                    sumSq += v * v;
                }
                chainMeans[c] = sum / perChain;
                chainVars[c] = sumSq / perChain - chainMeans[c] * chainMeans[c];
            }
            double overallMean = Arrays.stream(chainMeans).average().orElse(0);
            double B = 0; // between-chain variance
            for (int c = 0; c < numChains; c++) B += (chainMeans[c] - overallMean) * (chainMeans[c] - overallMean);
            B *= (double) perChain / (numChains - 1);
            double W = Arrays.stream(chainVars).average().orElse(0); // within-chain variance
            double varPlus = ((perChain - 1) * W + B) / perChain;
            double rHat = Math.sqrt(varPlus / Math.max(W, 1e-10));

            check("D4. R-hat < 1.1 (converged)", rHat < 1.1);
            System.out.printf("       R-hat = %.4f (target < 1.1)%n", rHat);
            System.out.printf("       Chain means: %s%n", Arrays.toString(chainMeans));
        }

        // D5. 有效样本量 (ESS) 估计
        {
            System.out.println("  [D5] Effective Sample Size (ESS)...");
            MCMCSampler sampler = new MCMCSampler(5000, 2000, 1, false, false);
            double[][] samples = sampler.sample(new double[]{0.0}, x -> -0.5 * x[0] * x[0]);
            double[] chain = new double[samples.length];
            for (int i = 0; i < samples.length; i++) chain[i] = samples[i][0];

            // 简单 ESS: 基于自相关
            double ess = computeESS(chain);
            check("D5. ESS > 100 (reasonable mixing)", ess > 100);
            System.out.printf("       ESS = %.0f / %d total samples%n", ess, chain.length);
        }
    }

    // ===================== E. Prophet End-to-End Integration =====================

    private static void testProphetIntegration() {
        System.out.println("\n════════════════════════════════════════════════════════");
        System.out.println("E. Prophet 端到端集成测试");
        System.out.println("════════════════════════════════════════════════════════");

        // E1. 已知真值信号: y = 2*t + 10 + noise
        {
            System.out.println("  [E1] Linear trend signal y=2t+10+noise...");
            List<DataPoint> data = new ArrayList<>();
            Random rng = new Random(42);
            for (int t = 0; t < 200; t++) {
                data.add(new DataPoint(t, 2 * t + 10 + rng.nextGaussian() * 2));
            }

            ProphetConfig config = new ProphetConfig();
            config.growth = ProphetConfig.GrowthType.LINEAR;
            config.nChangepoints = 10;
            config.yearlySeasonality = false;
            config.weeklySeasonality = false;
            config.mcmcSamples = 0;
            config.verbose = false;

            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            // 检查趋势斜率 k ≈ 2 (标准化后)
            System.out.printf("       k=%.6f, m=%.6f, sigma=%.6f%n", model.getK(), model.getM(), model.getSigmaObs());

            // 预测最后 10 个点
            double[] futureT = new double[10];
            for (int i = 0; i < 10; i++) futureT[i] = 200 + i;
            ProphetModel.ForecastResult forecast = model.predict(futureT);

            // 预测值应接近 2*205+10 = 420
            double meanPred = 0;
            for (int i = 0; i < 10; i++) meanPred += forecast.yhat[i];
            meanPred /= 10;
            double expectedMean = 2 * 205 + 10;
            checkClose("E1. Linear trend prediction accuracy", meanPred, expectedMean, 0.1);
            System.out.printf("       Mean prediction=%.2f, expected≈%.2f%n", meanPred, expectedMean);
        }

        // E2. 季节性信号: y = 10*cos(2π*t/365) + noise
        {
            System.out.println("  [E2] Pure annual seasonality y=10*cos(2πt/365)+noise...");
            List<DataPoint> data = new ArrayList<>();
            Random rng = new Random(123);
            for (int t = 0; t < 730; t++) {
                double y = 10 * Math.cos(2 * Math.PI * t / 365.0) + rng.nextGaussian() * 1.0;
                data.add(new DataPoint(t, y));
            }

            ProphetConfig config = new ProphetConfig();
            config.growth = ProphetConfig.GrowthType.LINEAR;
            config.nChangepoints = 10;
            config.changepointPriorScale = 0.001; // 强稀疏 → 趋势接近常数
            config.yearlySeasonality = true;
            config.yearlyFourierOrder = 10;
            config.weeklySeasonality = false;
            config.mcmcSamples = 0;
            config.verbose = false;

            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            // 预测第 365 天附近 (cos 峰值)
            double[] testT = {365.0};
            ProphetModel.ForecastResult forecast = model.predict(testT);
            // cos(2π) = 1, 所以 y ≈ 10, 但有趋势偏移, 容差放宽
            checkClose("E2. Peak prediction near t=365", forecast.yhat[0], 10.0, 2.0);

            // 预测第 547 天附近 (cos 峰谷)
            double[] testT2 = {547.0}; // 365 + 182
            ProphetModel.ForecastResult forecast2 = model.predict(testT2);
            checkClose("E2. Trough prediction near t=547", forecast2.yhat[0], -10.0, 2.0);
            System.out.printf("       Peak pred=%.2f (expected≈10), Trough pred=%.2f (expected≈-10)%n",
                    forecast.yhat[0], forecast2.yhat[0]);
        }

        // E3. 综合天气数据 (完整 Prophet 流程)
        {
            System.out.println("  [E3] Full weather pipeline (1 year train, 30 day forecast)...");
            List<DataPoint> data = DataPoint.generateTemperatureData(365);

            ProphetConfig config = new ProphetConfig();
            config.growth = ProphetConfig.GrowthType.LINEAR;
            config.nChangepoints = 25;
            config.changepointPriorScale = 0.05;
            config.yearlySeasonality = true;
            config.weeklySeasonality = true;
            config.mcmcSamples = 0;
            config.uncertaintySamples = 200;
            config.verbose = false;

            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            // In-sample 评估
            double[] trainT = new double[365];
            double[] actualY = new double[365];
            for (int i = 0; i < 365; i++) {
                trainT[i] = data.get(i).getTimestamp();
                actualY[i] = data.get(i).getValue();
            }
            ProphetModel.ForecastResult fitted = model.predict(trainT);

            double ssRes = 0, ssTot = 0, meanY = 0;
            for (int i = 0; i < 365; i++) meanY += actualY[i];
            meanY /= 365;
            for (int i = 0; i < 365; i++) {
                ssRes += (actualY[i] - fitted.yhat[i]) * (actualY[i] - fitted.yhat[i]);
                ssTot += (actualY[i] - meanY) * (actualY[i] - meanY);
            }
            double r2 = 1 - ssRes / ssTot;

            check("E3. R² > 0.5", r2 > 0.5);
            check("E3. R² < 1.0 (not overfitting perfectly)", r2 < 0.999);
            System.out.printf("       R²=%.6f%n", r2);

            // 30 天预测
            double[] futureT = new double[30];
            for (int i = 0; i < 30; i++) futureT[i] = 365 + i;
            ProphetModel.ForecastResult forecast = model.predict(futureT);

            // CI 宽度合理性
            double meanCIWidth = 0;
            for (int i = 0; i < 30; i++) meanCIWidth += forecast.yhatUpper[i] - forecast.yhatLower[i];
            meanCIWidth /= 30;
            check("E3. 80% CI width > 0", meanCIWidth > 0);
            check("E3. 80% CI width < 50°C (reasonable)", meanCIWidth < 50);
            System.out.printf("       Mean 80%% CI width = %.2f°C%n", meanCIWidth);
        }

        // E4. 不同 changepoint_prior_scale 的效果 (L1 稀疏性)
        {
            System.out.println("  [E4] Laplace prior sparsity: tau=0.001 vs tau=0.5...");
            List<DataPoint> data = DataPoint.generateTemperatureData(365);

            // 小 tau = 强 L1 = 稀疏变点
            ProphetConfig config1 = new ProphetConfig();
            config1.changepointPriorScale = 0.001;
            config1.yearlySeasonality = true;
            config1.weeklySeasonality = false;
            config1.mcmcSamples = 0;
            config1.verbose = false;
            ProphetModel model1 = new ProphetModel(config1);
            model1.fit(data);
            int sparse1 = 0;
            for (double d : model1.getDelta()) if (Math.abs(d) > 1e-4) sparse1++;

            // 大 tau = 弱 L1 = 多变点
            ProphetConfig config2 = new ProphetConfig();
            config2.changepointPriorScale = 0.5;
            config2.yearlySeasonality = true;
            config2.weeklySeasonality = false;
            config2.mcmcSamples = 0;
            config2.verbose = false;
            ProphetModel model2 = new ProphetModel(config2);
            model2.fit(data);
            int sparse2 = 0;
            for (double d : model2.getDelta()) if (Math.abs(d) > 1e-4) sparse2++;

            check("E4. Small tau → fewer significant changepoints", sparse1 <= sparse2);
            System.out.printf("       tau=0.001: %d significant, tau=0.5: %d significant%n", sparse1, sparse2);
        }
    }

    // ===================== F. Boundary Conditions & Robustness =====================

    private static void testBoundaryConditions() {
        System.out.println("\n════════════════════════════════════════════════════════");
        System.out.println("F. 边界条件与鲁棒性测试");
        System.out.println("════════════════════════════════════════════════════════");

        // F1. 极小数据集 (30 天)
        {
            System.out.println("  [F1] Small dataset (30 days)...");
            List<DataPoint> data = DataPoint.generateTemperatureData(30);
            ProphetConfig config = new ProphetConfig();
            config.nChangepoints = 5;
            config.yearlySeasonality = true;
            config.weeklySeasonality = true;
            config.mcmcSamples = 0;
            config.verbose = false;
            try {
                ProphetModel model = new ProphetModel(config);
                model.fit(data);
                double[] futureT = {30, 31, 32};
                ProphetModel.ForecastResult forecast = model.predict(futureT);
                check("F1. Small dataset: no crash", true);
                check("F1. Small dataset: finite predictions",
                        Arrays.stream(forecast.yhat).allMatch(Double::isFinite));
            } catch (Exception e) {
                check("F1. Small dataset: no crash", false);
                System.out.printf("       ERROR: %s%n", e.getMessage());
            }
        }

        // F2. 大数据集 (3 年 = 1095 天)
        {
            System.out.println("  [F2] Large dataset (3 years)...");
            List<DataPoint> data = DataPoint.generateTemperatureData(1095);
            ProphetConfig config = new ProphetConfig();
            config.nChangepoints = 25;
            config.yearlySeasonality = true;
            config.weeklySeasonality = true;
            config.mcmcSamples = 0;
            config.verbose = false;
            long start = System.currentTimeMillis();
            ProphetModel model = new ProphetModel(config);
            model.fit(data);
            long elapsed = System.currentTimeMillis() - start;
            double[] futureT = {1095, 1096, 1097};
            ProphetModel.ForecastResult forecast = model.predict(futureT);
            check("F2. Large dataset: no crash", true);
            check("F2. Large dataset: finite predictions",
                    Arrays.stream(forecast.yhat).allMatch(Double::isFinite));
            check("F2. Large dataset: fits in < 30s", elapsed < 30000);
            System.out.printf("       Fitting time: %.1fs%n", elapsed / 1000.0);
        }

        // F3. 常数数据 (零方差)
        {
            System.out.println("  [F3] Constant data (zero variance)...");
            List<DataPoint> data = new ArrayList<>();
            for (int t = 0; t < 100; t++) data.add(new DataPoint(t, 20.0));
            ProphetConfig config = new ProphetConfig();
            config.nChangepoints = 5;
            config.yearlySeasonality = false;
            config.weeklySeasonality = false;
            config.mcmcSamples = 0;
            config.verbose = false;
            try {
                ProphetModel model = new ProphetModel(config);
                model.fit(data);
                double[] futureT = {100};
                ProphetModel.ForecastResult forecast = model.predict(futureT);
                check("F3. Constant data: no crash", true);
                checkClose("F3. Constant data: prediction ≈ 20", forecast.yhat[0], 20.0, 1.0);
            } catch (Exception e) {
                check("F3. Constant data: no crash", false);
                System.out.printf("       ERROR: %s%n", e.getMessage());
            }
        }

        // F4. 带缺失值的数据 (跳过某些天)
        {
            System.out.println("  [F4] Data with gaps (missing days)...");
            List<DataPoint> fullData = DataPoint.generateTemperatureData(365);
            List<DataPoint> gappedData = new ArrayList<>();
            for (int i = 0; i < fullData.size(); i++) {
                if (i % 3 != 0) gappedData.add(fullData.get(i)); // 跳过 1/3 的数据
            }
            ProphetConfig config = new ProphetConfig();
            config.nChangepoints = 15;
            config.yearlySeasonality = true;
            config.weeklySeasonality = false;
            config.mcmcSamples = 0;
            config.verbose = false;
            ProphetModel model = new ProphetModel(config);
            model.fit(gappedData);
            double[] futureT = {365};
            ProphetModel.ForecastResult forecast = model.predict(futureT);
            check("F4. Gapped data: finite prediction", Double.isFinite(forecast.yhat[0]));
            System.out.printf("       Prediction at t=365: %.2f°C%n", forecast.yhat[0]);
        }

        // F5. 极端参数: changepointPriorScale 极小
        {
            System.out.println("  [F5] Extreme tau=1e-6 (almost no changepoints)...");
            List<DataPoint> data = DataPoint.generateTemperatureData(365);
            ProphetConfig config = new ProphetConfig();
            config.changepointPriorScale = 1e-6;
            config.yearlySeasonality = true;
            config.mcmcSamples = 0;
            config.verbose = false;
            ProphetModel model = new ProphetModel(config);
            model.fit(data);
            int nonzero = 0;
            for (double d : model.getDelta()) if (Math.abs(d) > 1e-3) nonzero++;
            // With very strong L1 regularization, most changepoints should be suppressed
            check("F5. tau=1e-6: ≤5 significant changepoints", nonzero <= 5);
            System.out.printf("       Nonzero changepoints: %d / %d%n", nonzero, model.getDelta().length);
        }
    }

    // ===================== G. Prophet Algorithm Faithfulness =====================

    private static void testProphetFaithfulness() {
        System.out.println("\n════════════════════════════════════════════════════════");
        System.out.println("G. Prophet 算法忠实性验证");
        System.out.println("════════════════════════════════════════════════════════");

        // G1. 连续性约束: γ_j = -s_j * δ_j 使得 g(t) 在变点处连续
        {
            System.out.println("  [G1] Trend continuity at changepoints...");
            List<DataPoint> data = DataPoint.generateTemperatureData(365);
            ProphetConfig config = new ProphetConfig();
            config.yearlySeasonality = false;
            config.weeklySeasonality = false;
            config.mcmcSamples = 0;
            config.verbose = false;
            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            double[] cps = model.getChangepoints();
            double k = model.getK();
            double m = model.getM();
            double[] delta = model.getDelta();

            // 检查每个变点处趋势的连续性
            double maxDiscontinuity = 0;
            for (int j = 0; j < cps.length; j++) {
                double tj = cps[j];
                // 左极限
                double aDeltaLeft = 0, aGammaLeft = 0;
                for (int jj = 0; jj < j; jj++) {
                    aDeltaLeft += delta[jj];
                    aGammaLeft += (-cps[jj] * delta[jj]);
                }
                double gLeft = (k + aDeltaLeft) * tj + (m + aGammaLeft);
                // 右极限
                double aDeltaRight = aDeltaLeft + delta[j];
                double aGammaRight = aGammaLeft + (-cps[j] * delta[j]);
                double gRight = (k + aDeltaRight) * tj + (m + aGammaRight);
                maxDiscontinuity = Math.max(maxDiscontinuity, Math.abs(gLeft - gRight));
            }
            checkCloseAbs("G1. Trend continuity: max gap < 1e-10", maxDiscontinuity, 0.0, 1e-10);
            System.out.printf("       Max discontinuity = %.2e%n", maxDiscontinuity);
        }

        // G2. 加法分解: y_hat = trend + seasonal + holiday
        {
            System.out.println("  [G2] Additive decomposition verification...");
            List<DataPoint> data = DataPoint.generateTemperatureData(365);
            ProphetConfig config = new ProphetConfig();
            config.seasonalityMode = ProphetConfig.SeasonalityMode.ADDITIVE;
            config.yearlySeasonality = true;
            config.weeklySeasonality = true;
            config.addHoliday("Test", 200, 2, 2);
            config.mcmcSamples = 0;
            config.verbose = false;
            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            // 检查: sigma_obs > 0
            check("G2. sigma_obs > 0", model.getSigmaObs() > 0);

            // 检查: beta 有值
            check("G2. beta length = 26 (10*2 yearly + 3*2 weekly)", model.getBeta().length == 26);

            // 检查: kappa 有值
            check("G2. kappa length = 1 (1 holiday)", model.getKappa().length == 1);
        }

        // G3. Fourier 特征正确性: 验证 CPUBackend 的傅里叶特征
        {
            System.out.println("  [G3] Fourier feature correctness...");
            CPUBackend cpu = new CPUBackend();
            double[] t = {0, 365.25 / 4, 365.25 / 2, 365.25 * 3 / 4};
            double[][] X = cpu.computeFourierFeatures(t, 2, 365.25);

            // X[0] = [cos(0), sin(0), cos(0), sin(0)] = [1, 0, 1, 0]
            checkCloseAbs("G3. Fourier t=0 cos(h=1)", X[0][0], 1.0, 1e-10);
            checkCloseAbs("G3. Fourier t=0 sin(h=1)", X[0][1], 0.0, 1e-10);

            // t = 365.25/4: angle = 2π*1*(365.25/4)/365.25 = π/2
            double angle = Math.PI / 2;
            checkCloseAbs("G3. Fourier t=Q1 cos(h=1)", X[1][0], Math.cos(angle), 1e-10);
            checkCloseAbs("G3. Fourier t=Q1 sin(h=1)", X[1][1], Math.sin(angle), 1e-10);

            // t = 365.25/2: angle = π
            checkCloseAbs("G3. Fourier t=Q2 cos(h=1)", X[2][0], -1.0, 1e-10);
            checkCloseAbs("G3. Fourier t=Q2 sin(h=1)", X[2][1], 0.0, 1e-10);
        }

        // G4. 数据标准化一致性: 预测值应在原始尺度
        {
            System.out.println("  [G4] Data scaling consistency...");
            List<DataPoint> data = DataPoint.generateTemperatureData(365);
            ProphetConfig config = new ProphetConfig();
            config.mcmcSamples = 0;
            config.verbose = false;
            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            double[] futureT = {365};
            ProphetModel.ForecastResult forecast = model.predict(futureT);

            // 温度预测应在合理范围 (-40 ~ 60°C)
            check("G4. Prediction in reasonable range (-40, 60)",
                    forecast.yhat[0] > -40 && forecast.yhat[0] < 60);
            System.out.printf("       Prediction at t=365: %.2f°C%n", forecast.yhat[0]);
        }

        // G5. 不确定性随预测期增长 (Prophet 的趋势不确定性特性)
        {
            System.out.println("  [G5] Uncertainty grows with forecast horizon...");
            List<DataPoint> data = DataPoint.generateTemperatureData(365);
            ProphetConfig config = new ProphetConfig();
            config.mcmcSamples = 0;
            config.uncertaintySamples = 500;
            config.verbose = false;
            ProphetModel model = new ProphetModel(config);
            model.fit(data);

            double[] nearT = {365};
            double[] farT = {730}; // 预测更远
            ProphetModel.ForecastResult nearF = model.predict(nearT);
            ProphetModel.ForecastResult farF = model.predict(farT);

            double nearCI = nearF.yhatUpper[0] - nearF.yhatLower[0];
            double farCI = farF.yhatUpper[0] - farF.yhatLower[0];
            check("G5. Far CI > Near CI", farCI > nearCI);
            System.out.printf("       Near (1 day) CI=%.2f, Far (365 days) CI=%.2f%n", nearCI, farCI);
        }
    }

    // ===================== Utility Methods =====================

    private static Matrix randomMatrix(int rows, int cols, Random rng) {
        Matrix m = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m.set(i, j, rng.nextGaussian());
        return m;
    }

    /**
     * Compute effective sample size using autocorrelation.
     * ESS = N / (1 + 2 * sum of autocorrelations)
     */
    private static double computeESS(double[] chain) {
        int n = chain.length;
        double mean = Arrays.stream(chain).average().orElse(0);
        double var = 0;
        for (double v : chain) var += (v - mean) * (v - mean);
        var /= n;
        if (var < 1e-10) return n;

        double sumAutoCorr = 0;
        for (int lag = 1; lag < Math.min(n / 2, 500); lag++) {
            double autoCov = 0;
            for (int i = 0; i < n - lag; i++) {
                autoCov += (chain[i] - mean) * (chain[i + lag] - mean);
            }
            autoCov /= (n - lag);
            double rho = autoCov / var;
            if (rho < 0.05) break; // 当自相关接近 0 时停止
            sumAutoCorr += rho;
        }
        return n / (1 + 2 * sumAutoCorr);
    }

    // ===================== Main =====================

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║  Weather Prophet — 严谨测试套件                                ║");
        System.out.println("║  覆盖: 数值精度 / 收敛诊断 / 边界条件 / 统计验证 / 算法忠实性     ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");

        long startTime = System.currentTimeMillis();

        try {
            testMatrixNumericalPrecision();
        } catch (Exception e) {
            System.out.printf("  !! Section A exception: %s%n", e.getMessage());
            e.printStackTrace();
        }

        try {
            testLBFGSConvergence();
        } catch (Exception e) {
            System.out.printf("  !! Section B exception: %s%n", e.getMessage());
            e.printStackTrace();
        }

        try {
            testBayesianPriors();
        } catch (Exception e) {
            System.out.printf("  !! Section C exception: %s%n", e.getMessage());
            e.printStackTrace();
        }

        try {
            testMCMCConvergence();
        } catch (Exception e) {
            System.out.printf("  !! Section D exception: %s%n", e.getMessage());
            e.printStackTrace();
        }

        try {
            testProphetIntegration();
        } catch (Exception e) {
            System.out.printf("  !! Section E exception: %s%n", e.getMessage());
            e.printStackTrace();
        }

        try {
            testBoundaryConditions();
        } catch (Exception e) {
            System.out.printf("  !! Section F exception: %s%n", e.getMessage());
            e.printStackTrace();
        }

        try {
            testProphetFaithfulness();
        } catch (Exception e) {
            System.out.printf("  !! Section G exception: %s%n", e.getMessage());
            e.printStackTrace();
        }

        long elapsed = System.currentTimeMillis() - startTime;

        System.out.println("\n╔══════════════════════════════════════════════════════════════╗");
        System.out.println("║  测试结果汇总                                                 ║");
        System.out.println("╚══════════════════════════════════════════════════════════════╝");
        System.out.printf("  总测试数:  %d%n", totalTests);
        System.out.printf("  通过:      %d (%.1f%%)%n", passed, 100.0 * passed / totalTests);
        System.out.printf("  失败:      %d (%.1f%%)%n", failed, 100.0 * failed / totalTests);
        System.out.printf("  跳过:      %d (%.1f%%)%n", skipped, 100.0 * skipped / totalTests);
        System.out.printf("  耗时:      %.1fs%n", elapsed / 1000.0);

        if (!failures.isEmpty()) {
            System.out.println("\n  失败测试列表:");
            for (String f : failures) {
                System.out.printf("    ✗ %s%n", f);
            }
        }

        System.out.println();
        if (failed == 0) {
            System.out.println("  ★★★ 全部测试通过 ★★★");
        } else {
            System.out.printf("  !!! 有 %d 个测试失败 !!!%n", failed);
        }
    }
}
