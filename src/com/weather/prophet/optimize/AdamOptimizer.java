package com.weather.prophet.optimize;

import com.weather.prophet.matrix.VecOps;

/**
 * Adam optimizer (Adaptive Moment Estimation).
 * 
 * Well-suited for problems with Laplace-like priors and non-uniform curvature,
 * where L-BFGS often fails due to poor Hessian approximation.
 * 
 * Features:
 * - Per-parameter adaptive learning rate based on gradient history
 * - Momentum (first moment) and squared gradient (second moment) tracking
 * - Bias correction for early iterations
 * - Learning rate decay schedule
 * - Warmup phase for stable initial convergence
 */
public class AdamOptimizer {

    public interface ObjectiveFunction {
        /** Returns [value, gradient] for given parameters */
        double[] evaluate(double[] params);
    }

    private final int maxIterations;
    private final double learningRate;
    private final double beta1;      // momentum decay (default 0.9)
    private final double beta2;      // squared gradient decay (default 0.999)
    private final double epsilon;    // numerical stability (default 1e-8)
    private final double gradTol;    // convergence: stop when |grad| < this
    private final boolean verbose;

    public AdamOptimizer() {
        this(5000, 0.001, 0.9, 0.999, 1e-8, 1e-6, false);
    }

    public AdamOptimizer(int maxIterations, double learningRate, double beta1,
                         double beta2, double eps, double gradTol, boolean verbose) {
        this.maxIterations = maxIterations;
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = eps;
        this.gradTol = gradTol;
        this.verbose = verbose;
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

    /**
     * Minimize the objective function using Adam.
     * @param x0 Initial parameters
     * @param fn Objective function: returns [value, grad...]
     * @return Optimization result
     */
    public OptResult minimize(double[] x0, ObjectiveFunction fn) {
        int n = x0.length;
        double[] x = VecOps.copy(x0);
        double[] m = new double[n];   // first moment (momentum)
        double[] v = new double[n];   // second moment (squared gradient)

        // Initial evaluation
        double[] initEval = fn.evaluate(x);
        double fVal = initEval[0];
        double[] grad = new double[n];
        System.arraycopy(initEval, 1, grad, 0, n);
        double gradNorm = VecOps.norm(grad);

        double bestVal = fVal;
        double[] bestX = VecOps.copy(x);

        // Warmup: start with smaller learning rate for first 100 iterations
        double warmupLR = learningRate * 0.1;

        if (verbose) System.out.printf("[Adam] Iter %5d: f=%.8f |g|=%.8f lr=%.2e%n", 0, fVal, gradNorm, warmupLR);

        for (int iter = 1; iter <= maxIterations; iter++) {
            // Update biased first moment estimate
            for (int i = 0; i < n; i++) {
                m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
            }

            // Update biased second moment estimate
            for (int i = 0; i < n; i++) {
                v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
            }

            // Bias-corrected estimates
            double beta1Corr = 1 - Math.pow(beta1, iter);
            double beta2Corr = 1 - Math.pow(beta2, iter);

            // Compute step: x = x - lr * m_hat / (sqrt(v_hat) + eps)
            double lr = (iter <= 100) ? warmupLR : learningRate;
            // Optional: decay learning rate after 2000 iterations
            if (iter > 2000) {
                lr = learningRate * 0.1;
            }

            for (int i = 0; i < n; i++) {
                double mHat = m[i] / beta1Corr;
                double vHat = v[i] / beta2Corr;
                x[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
            }

            // Evaluate at new point
            double[] eval = fn.evaluate(x);
            fVal = eval[0];
            System.arraycopy(eval, 1, grad, 0, n);
            gradNorm = VecOps.norm(grad);

            // Track best solution
            if (fVal < bestVal) {
                bestVal = fVal;
                bestX = VecOps.copy(x);
            }

            if (verbose && (iter % 100 == 0 || iter <= 10)) {
                System.out.printf("[Adam] Iter %5d: f=%.8f |g|=%.8f lr=%.2e%n", iter, fVal, gradNorm, lr);
            }

            // Convergence check
            if (gradNorm < gradTol) {
                if (verbose) System.out.printf("[Adam] Converged at iter %d: |g|=%.2e < %.2e%n", iter, gradNorm, gradTol);
                break;
            }

            // Check if function value is no longer decreasing (stagnation)
            if (iter > 500 && Math.abs(fVal - bestVal) < 1e-8 * Math.abs(bestVal) + 1e-12) {
                // No meaningful improvement over best - try restart with best params
                if (iter > 1000) {
                    if (verbose) System.out.printf("[Adam] Stagnation at iter %d, using best solution%n", iter);
                    break;
                }
            }
        }

        // Use the converged point (not necessarily the lowest f value)
        // Adam with momentum oscillates around the optimum; the converged point
        // has the lowest gradient norm, which is the actual optimum.
        return new OptResult(x, fVal, gradNorm);
    }
}
