package com.weather.prophet.optimize;

import com.weather.prophet.matrix.VecOps;

/**
 * L-BFGS optimizer with robust line search and stall escape.
 *
 * Key improvements:
 * - Adaptive initial step size based on gradient norm
 * - Armijo-only line search (no Wolfe, which caused step collapse)
 * - Quadratic interpolation for faster step reduction
 * - Stall escape: reset history + steepest descent fallback
 */
public class LBFGSOptimizer {

    public interface ObjectiveFunction {
        double[] evaluate(double[] params);
    }

    private final int m;
    private final int maxIterations;
    private final double gradTol;
    private final double paramTol;
    private final double funcTol;
    private final boolean verbose;

    public LBFGSOptimizer() {
        this(7, 10000, 1e-6, 1e-10, 1e-10, false);
    }

    public LBFGSOptimizer(int m, int maxIter, double gradTol,
                          double paramTol, double funcTol, boolean verbose) {
        this.m = m;
        this.maxIterations = maxIter;
        this.gradTol = gradTol;
        this.paramTol = paramTol;
        this.funcTol = funcTol;
        this.verbose = verbose;
    }

    public OptResult minimize(double[] x0, ObjectiveFunction fn) {
        int n = x0.length;
        double[] x = VecOps.copy(x0);
        double[] grad, xNew, gradNew;
        double fVal, fNew;

        double[] initEval = fn.evaluate(x);
        fVal = initEval[0];
        grad = new double[n];
        System.arraycopy(initEval, 1, grad, 0, n);

        double[][] sList = new double[m][n];
        double[][] yList = new double[m][n];
        double[] rhoList = new double[m];
        int histSize = 0, histPtr = 0;

        double[] prevX = VecOps.copy(x);
        double prevVal = fVal;
        double prevStepSize = -1.0;
        int noProgressCount = 0;
        int stallResets = 0;
        final int MAX_STALL_RESETS = 5;

        if (verbose) System.out.printf("[L-BFGS] Iter %5d: f=%.8f |g|=%.8f%n",
            0, fVal, VecOps.norm(grad));

        for (int iter = 1; iter <= maxIterations; iter++) {
            // ─── L-BFGS two-loop recursion ───
            double[] q = VecOps.copy(grad);
            double[] alpha = new double[m];
            int count = Math.min(histSize, m);

            for (int i = count - 1; i >= 0; i--) {
                int idx = (histPtr - count + i + m) % m;
                alpha[idx] = rhoList[idx] * VecOps.dot(sList[idx], q);
                q = VecOps.subtract(q, VecOps.scale(yList[idx], alpha[idx]));
            }

            double gamma = 1.0;
            if (histSize > 0) {
                int lastIdx = (histPtr - 1 + m) % m;
                double ys = VecOps.dot(yList[lastIdx], sList[lastIdx]);
                double yy = VecOps.dot(yList[lastIdx], yList[lastIdx]);
                if (ys > 1e-20) gamma = ys / yy;
            }

            double[] r = VecOps.scale(q, gamma);
            for (int i = 0; i < count; i++) {
                int idx = (histPtr - count + i + m) % m;
                double beta = rhoList[idx] * VecOps.dot(yList[idx], r);
                r = VecOps.add(r, VecOps.scale(sList[idx], alpha[idx] - beta));
            }

            double[] direction = VecOps.scale(r, -1.0);
            double dirDeriv = VecOps.dot(grad, direction);

            if (dirDeriv >= 0) {
                direction = VecOps.scale(grad, -1.0);
                dirDeriv = -VecOps.dot(grad, grad);
                histSize = 0;
                histPtr = 0;
            }

            // ─── Initial step size ───
            double stepSize;
            if (prevStepSize > 1e-15) {
                stepSize = Math.min(2.0 * prevStepSize, 1.0);
                stepSize = Math.max(stepSize, 1e-10);
            } else {
                double absF = Math.abs(fVal) + 1.0;
                stepSize = Math.min(1.0, 0.01 * absF / (1e-4 * (Math.abs(dirDeriv) + 1e-10)));
                stepSize = Math.max(stepSize, 1e-10);
            }

            // ─── Line search (Armijo only) ───
            double c1 = 1e-4;
            boolean lsOk = false;
            fNew = fVal;
            gradNew = grad;
            xNew = x;

            for (int ls = 0; ls < 60; ls++) {
                xNew = VecOps.add(x, VecOps.scale(direction, stepSize));
                double[] evalNew = fn.evaluate(xNew);
                fNew = evalNew[0];
                gradNew = new double[n];
                System.arraycopy(evalNew, 1, gradNew, 0, n);

                if (fNew <= fVal + c1 * stepSize * dirDeriv) {
                    lsOk = true;
                    break;
                }

                // Quadratic interpolation on first failure
                if (ls == 0 && fNew < Double.MAX_VALUE / 2) {
                    double denom = 2.0 * (fNew - fVal - dirDeriv * stepSize);
                    if (denom > 1e-10) {
                        double quadStep = -dirDeriv * stepSize / denom;
                        if (quadStep > 1e-10 && quadStep < 0.9 * stepSize) {
                            stepSize = quadStep;
                            continue;
                        }
                    }
                }
                stepSize *= 0.5;
            }

            // ─── Fallback: steepest descent with small step ───
            if (!lsOk) {
                direction = VecOps.scale(grad, -1.0);
                dirDeriv = -VecOps.dot(grad, grad);
                stepSize = Math.min(1.0 / (VecOps.norm(grad) + 1.0), 1e-2);
                xNew = VecOps.add(x, VecOps.scale(direction, stepSize));
                double[] evalNew = fn.evaluate(xNew);
                fNew = evalNew[0];
                gradNew = new double[n];
                System.arraycopy(evalNew, 1, gradNew, 0, n);

                if (fNew < fVal) {
                    lsOk = true;
                } else {
                    if (verbose) System.out.println("[L-BFGS] Line search failed, stopping.");
                    break;
                }
            }

            // ─── Update L-BFGS history ───
            double[] s = VecOps.subtract(xNew, x);
            double[] y = VecOps.subtract(gradNew, grad);
            double sy = VecOps.dot(s, y);

            if (sy > 1e-20) {
                int storeIdx = histPtr % m;
                sList[storeIdx] = s;
                yList[storeIdx] = y;
                rhoList[storeIdx] = 1.0 / sy;
                histPtr++;
                histSize = Math.min(histSize + 1, m);
            }

            // ─── Progress check & stall escape ───
            double fChange = fVal - fNew;  // positive means improvement
            double paramChange = VecOps.norm(VecOps.subtract(xNew, x));
            double gradNorm = VecOps.norm(gradNew);

            if (fChange < funcTol * (Math.abs(fVal) + 1.0) || paramChange < paramTol) {
                noProgressCount++;
            } else {
                noProgressCount = 0;
            }

            // Stall escape: reset and try steepest descent
            if (noProgressCount > 20 && stallResets < MAX_STALL_RESETS) {
                histSize = 0;
                histPtr = 0;
                prevStepSize = -1.0;
                stallResets++;
                noProgressCount = 0;
                if (verbose) System.out.printf("[L-BFGS] Stall escape #%d at iter %d (|g|=%.2e)%n",
                    stallResets, iter, gradNorm);
            }

            prevX = VecOps.copy(x);
            prevVal = fVal;
            prevStepSize = stepSize;

            x = xNew;
            grad = gradNew;
            fVal = fNew;

            if (verbose && (iter % 100 == 0 || iter <= 10)) {
                System.out.printf("[L-BFGS] Iter %5d: f=%.8f |g|=%.2e step=%.2e%n",
                    iter, fVal, gradNorm, stepSize);
            }

            // ─── Convergence ───
            if (gradNorm < gradTol) {
                if (verbose) System.out.printf("[L-BFGS] Converged: |g|=%.2e%n", gradNorm);
                break;
            }

            // Stall convergence: after exhausting stall resets, accept solution
            // if gradient is small enough (within 1000x of tolerance)
            if (stallResets >= MAX_STALL_RESETS && gradNorm < gradTol * 1e3) {
                if (verbose) System.out.printf("[L-BFGS] Converged (stalled, |g|=%.2e)%n", gradNorm);
                break;
            }

            // No substantial progress for too long
            if (noProgressCount > 200) {
                if (verbose) System.out.printf("[L-BFGS] Converged (no progress, |g|=%.2e)%n", gradNorm);
                break;
            }
        }

        if (verbose) System.out.printf("[L-BFGS] Final: f=%.8f |g|=%.2e%n", fVal, VecOps.norm(grad));
        return new OptResult(x, fVal, VecOps.norm(grad));
    }

    public static class OptResult {
        public final double[] params;
        public final double value;
        public final double gradNorm;

        public OptResult(double[] params, double value, double gradNorm) {
            this.params = params;
            this.value = value;
            this.gradNorm = gradNorm;
        }
    }
}
