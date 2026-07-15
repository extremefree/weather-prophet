package com.weather.prophet.optimize;

import com.weather.prophet.matrix.VecOps;

/**
 * L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) optimizer.
 *
 * This is the same optimization algorithm used by Stan (via its L-BFGS implementation)
 * for Prophet's MAP (Maximum A Posteriori) estimation.
 *
 * L-BFGS approximates the Hessian using the last m gradient differences,
 * avoiding the need to store the full Hessian matrix — critical for large models.
 *
 * Features:
 * - Wolfe line search with backtracking
 * - Limited memory (default m=7 corrections)
 * - Convergence checks (gradient norm, parameter change, objective change)
 * - Max iterations safeguard
 */
public class LBFGSOptimizer {

    public interface ObjectiveFunction {
        /** Returns [value, gradient] for given parameters */
        double[] evaluate(double[] params);
    }

    private final int m;              // number of corrections to store
    private final int maxIterations;
    private final double gradTol;     // gradient tolerance
    private final double paramTol;    // parameter change tolerance
    private final double funcTol;     // function value change tolerance
    private final boolean verbose;

    public LBFGSOptimizer() {
        this(7, 10000, 1e-8, 1e-10, 1e-10, false);
    }

    public LBFGSOptimizer(int m, int maxIterations, double gradTol,
                          double paramTol, double funcTol, boolean verbose) {
        this.m = m;
        this.maxIterations = maxIterations;
        this.gradTol = gradTol;
        this.paramTol = paramTol;
        this.funcTol = funcTol;
        this.verbose = verbose;
    }

    /**
     * Minimize the objective function using L-BFGS.
     *
     * @param x0 initial parameter values
     * @param fn  objective function returning [value, gradient]
     * @return optimization result
     */
    public OptResult minimize(double[] x0, ObjectiveFunction fn) {
        int n = x0.length;
        double[] x = VecOps.copy(x0);
        double[] grad;
        double fVal;

        // Initial evaluation
        double[] initEval = fn.evaluate(x);
        fVal = initEval[0];
        grad = new double[n];
        System.arraycopy(initEval, 1, grad, 0, n);

        // L-BFGS storage
        double[][] sList = new double[m][n];  // parameter differences
        double[][] yList = new double[m][n];  // gradient differences
        double[] rhoList = new double[m];     // 1 / (y^T s)
        int histSize = 0;
        int histPtr = 0;

        double[] prevX = VecOps.copy(x);
        double[] prevGrad = VecOps.copy(grad);
        double prevVal = fVal;

        if (verbose) System.out.printf("[L-BFGS] Iter %5d: f=%.8f |g|=%.8f%n", 0, fVal, VecOps.norm(grad));

        for (int iter = 1; iter <= maxIterations; iter++) {
            // Compute search direction using two-loop recursion
            double[] q = VecOps.copy(grad);
            double[] alpha = new double[m];

            // First loop: go backward through history
            int count = Math.min(histSize, m);
            for (int i = count - 1; i >= 0; i--) {
                int idx = (histPtr - count + i + m) % m;
                alpha[idx] = rhoList[idx] * VecOps.dot(sList[idx], q);
                q = VecOps.subtract(q, VecOps.scale(yList[idx], alpha[idx]));
            }

            // Initial Hessian approximation: H0 = gamma * I
            double gamma = 1.0;
            if (histSize > 0) {
                int lastIdx = (histPtr - 1 + m) % m;
                double ys = VecOps.dot(yList[lastIdx], sList[lastIdx]);
                double yy = VecOps.dot(yList[lastIdx], yList[lastIdx]);
                if (ys > 1e-20) gamma = ys / yy;
            }
            double[] r = VecOps.scale(q, gamma);

            // Second loop: go forward through history
            for (int i = 0; i < count; i++) {
                int idx = (histPtr - count + i + m) % m;
                double beta = rhoList[idx] * VecOps.dot(yList[idx], r);
                r = VecOps.add(r, VecOps.scale(sList[idx], alpha[idx] - beta));
            }

            // r is now the search direction (negative of L-BFGS approximated Newton direction)
            // We want to minimize, so we move in the negative gradient direction
            // r = -H * grad, so direction = -r (but actually L-BFGS gives us H*grad,
            // and we want to go in direction -H*grad, so the step is x = x - step * H*grad
            // Since r = H*grad already, we just negate it
            double[] direction = VecOps.scale(r, -1.0);

            // Line search with Wolfe conditions (backtracking Armijo)
            double stepSize = 1.0;
            double c1 = 1e-4;  // Armijo constant
            double c2 = 0.9;   // curvature condition constant
            boolean lineSearchOk = false;

            for (int lsIter = 0; lsIter < 40; lsIter++) {
                double[] xNew = VecOps.add(x, VecOps.scale(direction, stepSize));
                double[] evalNew = fn.evaluate(xNew);
                double fNew = evalNew[0];
                double[] gradNew = new double[n];
                System.arraycopy(evalNew, 1, gradNew, 0, n);

                // Armijo condition: f(x + α*d) ≤ f(x) + c1 * α * g^T * d
                double armijo = fVal + c1 * stepSize * VecOps.dot(grad, direction);
                if (fNew <= armijo) {
                    // Check strong Wolfe condition: |g_new^T * d| ≤ c2 * |g^T * d|
                    double dirDerivNew = VecOps.dot(gradNew, direction);
                    double dirDerivOld = VecOps.dot(grad, direction);
                    if (Math.abs(dirDerivNew) <= c2 * Math.abs(dirDerivOld) || lsIter > 20) {
                        prevX = VecOps.copy(x);
                        prevGrad = VecOps.copy(grad);
                        prevVal = fVal;

                        // Update s and y
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

                        x = xNew;
                        grad = gradNew;
                        fVal = fNew;
                        lineSearchOk = true;
                        break;
                    }
                }
                stepSize *= 0.5;
            }

            if (!lineSearchOk) {
                if (verbose) System.out.println("[L-BFGS] Line search failed, stopping.");
                break;
            }

            if (verbose && iter % 100 == 0) {
                System.out.printf("[L-BFGS] Iter %5d: f=%.8f |g|=%.8f step=%.6f%n",
                        iter, fVal, VecOps.norm(grad), stepSize);
            }

            // Convergence checks
            double gradNorm = VecOps.norm(grad);
            if (gradNorm < gradTol) {
                if (verbose) System.out.printf("[L-BFGS] Converged: |grad|=%.2e < %.2e%n", gradNorm, gradTol);
                break;
            }

            double paramChange = VecOps.norm(VecOps.subtract(x, prevX));
            if (iter > 1 && paramChange < paramTol) {
                if (verbose) System.out.printf("[L-BFGS] Converged: |Δx|=%.2e < %.2e%n", paramChange, paramTol);
                break;
            }

            if (iter > 1 && Math.abs(fVal - prevVal) < funcTol * Math.abs(prevVal + 1e-10)) {
                if (verbose) System.out.printf("[L-BFGS] Converged: |Δf|=%.2e < %.2e*|f|%n",
                        Math.abs(fVal - prevVal), funcTol);
                break;
            }
        }

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
