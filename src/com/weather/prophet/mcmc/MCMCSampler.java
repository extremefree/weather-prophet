package com.weather.prophet.mcmc;

import com.weather.prophet.matrix.VecOps;
import com.weather.prophet.optimize.BayesianPriors;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * MCMC sampler for Prophet's Bayesian posterior inference.
 *
 * Prophet uses Stan's NUTS (No-U-Turn Sampler) for posterior sampling.
 * NUTS is a variant of Hamiltonian Monte Carlo (HMC) that automatically
 * tunes the number of leapfrog steps.
 *
 * This implementation provides:
 * 1. Metropolis-Hastings (baseline, always works)
 * 2. Simplified NUTS (Hamiltonian Monte Carlo with adaptive step size)
 *
 * MCMC samples are used for:
 * - Posterior predictive uncertainty intervals
 * - Capturing parameter correlations that MAP estimation misses
 */
public class MCMCSampler {

    public interface LogPosteriorFn {
        /** Returns log p(theta | data) up to a constant */
        double evaluate(double[] theta);
    }

    private final int numSamples;
    private final int numWarmup;
    private final int numChains;
    private final boolean useNUTS;
    private final boolean verbose;
    private final Random rng;

    public MCMCSampler(int numSamples, int numWarmup, int numChains, boolean useNUTS, boolean verbose) {
        this.numSamples = numSamples;
        this.numWarmup = numWarmup;
        this.numChains = numChains;
        this.useNUTS = useNUTS;
        this.verbose = verbose;
        this.rng = new Random(42);
    }

    /**
     * Run MCMC sampling from the MAP estimate.
     *
     * @param mapParams MAP parameter estimate (starting point)
     * @param logPostFn function computing log posterior
     * @return array of posterior samples [numSamples * numChains][numParams]
     */
    public double[][] sample(double[] mapParams, LogPosteriorFn logPostFn) {
        List<double[]> allSamples = new ArrayList<>();

        for (int chain = 0; chain < numChains; chain++) {
            if (verbose) System.out.printf("[MCMC] Starting chain %d...%n", chain);

            double[] current = VecOps.copy(mapParams);
            // Add small perturbation for different chains
            for (int i = 0; i < current.length; i++) {
                current[i] += rng.nextGaussian() * 0.01;
            }

            double currentLogPost = logPostFn.evaluate(current);
            if (Double.isNaN(currentLogPost) || Double.isInfinite(currentLogPost)) {
                currentLogPost = -1e20;
            }

            // Adaptive step size (dual averaging like Stan)
            double stepSize = 0.1;
            double targetAcceptRate = 0.8; // Stan's default for NUTS

            List<double[]> chainSamples = new ArrayList<>();
            int accepted = 0;

            for (int iter = 0; iter < numWarmup + numSamples; iter++) {
                double[] proposal;
                double proposalLogPost;

                if (useNUTS) {
                    // Simplified NUTS (HMC with adaptive tree building)
                    NUTSResult nutsResult = nutsStep(current, currentLogPost, logPostFn, stepSize, 10);
                    proposal = nutsResult.params;
                    proposalLogPost = nutsResult.logPost;
                    stepSize = adaptStepSize(stepSize, nutsResult.acceptRate, targetAcceptRate, iter);
                } else {
                    // Metropolis-Hastings with Gaussian proposal
                    proposal = new double[current.length];
                    for (int i = 0; i < current.length; i++) {
                        proposal[i] = current[i] + rng.nextGaussian() * stepSize;
                    }
                    proposalLogPost = logPostFn.evaluate(proposal);
                }

                // Accept/reject
                if (!Double.isNaN(proposalLogPost) && !Double.isInfinite(proposalLogPost)) {
                    double logAlpha = proposalLogPost - currentLogPost;
                    if (Math.log(rng.nextDouble()) < logAlpha) {
                        current = proposal;
                        currentLogPost = proposalLogPost;
                        accepted++;
                    }
                }

                // Collect sample (after warmup)
                if (iter >= numWarmup) {
                    chainSamples.add(VecOps.copy(current));
                }

                // Adapt step size during warmup
                if (!useNUTS && iter < numWarmup && iter > 0 && iter % 50 == 0) {
                    double acceptRate = (double) accepted / (iter + 1);
                    stepSize = adaptStepSize(stepSize, acceptRate, targetAcceptRate, iter);
                }

                if (verbose && iter % 200 == 0) {
                    double acceptRate = (double) accepted / (iter + 1);
                    System.out.printf("[MCMC] Chain %d, Iter %d: logPost=%.4f, acceptRate=%.3f, stepSize=%.6f%n",
                            chain, iter, currentLogPost, acceptRate, stepSize);
                }
            }

            allSamples.addAll(chainSamples);
            if (verbose) {
                double finalAcceptRate = (double) accepted / (numWarmup + numSamples);
                System.out.printf("[MCMC] Chain %d complete: final accept rate=%.3f%n", chain, finalAcceptRate);
            }
        }

        return allSamples.toArray(new double[0][]);
    }

    /**
     * NUTS (No-U-Turn Sampler) step - simplified version of Stan's NUTS.
     * Uses leapfrog integration in Hamiltonian dynamics.
     */
    private NUTSResult nutsStep(double[] current, double currentLogPost,
                                 LogPosteriorFn logPostFn, double stepSize, int maxDepth) {
        int n = current.length;

        // Sample momentum
        double[] momentum = new double[n];
        for (int i = 0; i < n; i++) momentum[i] = rng.nextGaussian();

        double currentKinetic = 0.5 * VecOps.dot(momentum, momentum);
        double currentHam = -currentLogPost + currentKinetic;

        // Leapfrog integration
        double[] q = VecOps.copy(current);
        double[] p = VecOps.copy(momentum);

        // Numerical gradient via finite differences
        double eps = stepSize;

        // Half step for momentum
        double[] gradQ = numericalGradient(q, logPostFn);
        for (int i = 0; i < n; i++) p[i] += 0.5 * eps * gradQ[i];

        // Full steps
        int numSteps = Math.min(1 << maxDepth, 4); // cap at 4 leapfrog steps for performance
        double bestLogPost = currentLogPost;
        double[] bestParams = VecOps.copy(current);
        double sumAcceptProb = 0;
        int validSteps = 0;

        for (int step = 0; step < numSteps; step++) {
            // Full step for position
            for (int i = 0; i < n; i++) q[i] += eps * p[i];

            // Full step for momentum (except last)
            gradQ = numericalGradient(q, logPostFn);
            if (step < numSteps - 1) {
                for (int i = 0; i < n; i++) p[i] += eps * gradQ[i];
            } else {
                // Half step for momentum at end
                for (int i = 0; i < n; i++) p[i] += 0.5 * eps * gradQ[i];
            }

            double qLogPost = logPostFn.evaluate(q);
            if (!Double.isNaN(qLogPost) && !Double.isInfinite(qLogPost)) {
                double qKinetic = 0.5 * VecOps.dot(p, p);
                double qHam = -qLogPost + qKinetic;
                double logAcceptProb = currentHam - qHam;
                sumAcceptProb += Math.min(1.0, Math.exp(logAcceptProb));
                validSteps++;

                if (logAcceptProb > 0 || Math.log(rng.nextDouble()) < logAcceptProb) {
                    bestParams = VecOps.copy(q);
                    bestLogPost = qLogPost;
                }

                // U-turn check (simplified)
                if (VecOps.dot(VecOps.subtract(q, current), p) < 0) {
                    break; // trajectory is turning around, stop
                }
            }
        }

        double avgAcceptRate = validSteps > 0 ? sumAcceptProb / validSteps : 0;
        return new NUTSResult(bestParams, bestLogPost, avgAcceptRate);
    }

    /**
     * Numerical gradient via central finite differences.
     * (Stan computes this analytically via autodiff, but this is faithful in principle.)
     */
    private double[] numericalGradient(double[] params, LogPosteriorFn fn) {
        double[] grad = new double[params.length];
        double h = 1e-5;
        for (int i = 0; i < params.length; i++) {
            double orig = params[i];
            params[i] = orig + h;
            double fp = fn.evaluate(params);
            params[i] = orig - h;
            double fm = fn.evaluate(params);
            params[i] = orig;
            grad[i] = (fp - fm) / (2 * h);
            if (Double.isNaN(grad[i])) grad[i] = 0;
        }
        return grad;
    }

    /**
     * Dual-averaging step size adaptation (like Stan).
     */
    private double adaptStepSize(double currentStep, double acceptRate,
                                  double targetAccept, int iteration) {
        // Simple heuristic: increase if too high acceptance, decrease if too low
        double ratio = acceptRate / targetAccept;
        if (ratio > 1.2) return currentStep * 1.1;
        if (ratio < 0.8) return currentStep * 0.9;
        return currentStep;
    }

    private static class NUTSResult {
        final double[] params;
        final double logPost;
        final double acceptRate;

        NUTSResult(double[] params, double logPost, double acceptRate) {
            this.params = params;
            this.logPost = logPost;
            this.acceptRate = acceptRate;
        }
    }
}
