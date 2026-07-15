package com.weather.prophet.compute;

/**
 * Compute backend interface for Prophet model.
 * Supports both CPU and GPU (OpenCL) computation.
 * GPU acceleration is critical for:
 * 1. Large-scale matrix operations (X^T X, Cholesky)
 * 2. Parallel MCMC chain sampling
 * 3. Fourier feature computation on large datasets
 */
public interface ComputeBackend {

    /** Backend type */
    enum Type { CPU, GPU }

    Type getType();

    /** Matrix multiply: C = A * B */
    double[][] matmul(double[][] A, double[][] B);

    /** Element-wise vector add: r = a + b */
    double[] vecAdd(double[] a, double[] b);

    /** Element-wise vector multiply: r = a * b */
    double[] vecMul(double[] a, double[] b);

    /** Dot product */
    double dot(double[] a, double[] b);

    /** Scale vector: r = a * s */
    double[] vecScale(double[] a, double s);

    /** Compute Fourier features in parallel: result[i][j] = cos/sin(2π * (j/2+1) * t[i] / period) */
    double[][] computeFourierFeatures(double[] t, int fourierOrder, double period);

    /** Check if GPU is available */
    boolean isAvailable();

    /** Release GPU resources */
    void release();
}
