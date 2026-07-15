package com.weather.prophet.matrix;

/**
 * Vector operations utility for Prophet optimization and MCMC.
 */
public class VecOps {

    public static double[] add(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
    }

    public static double[] subtract(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] - b[i];
        return r;
    }

    public static double[] scale(double[] a, double s) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] * s;
        return r;
    }

    public static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    public static double norm(double[] a) {
        return Math.sqrt(dot(a, a));
    }

    public static double[] copy(double[] a) {
        return a.clone();
    }

    public static void addInPlace(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++) a[i] += b[i];
    }

    public static void scaleInPlace(double[] a, double s) {
        for (int i = 0; i < a.length; i++) a[i] *= s;
    }

    /** Element-wise multiply */
    public static double[] elemMultiply(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] * b[i];
        return r;
    }

    /** Soft threshold for L1 proximal operator (used in Laplace prior) */
    public static double softThreshold(double x, double threshold) {
        if (x > threshold) return x - threshold;
        if (x < -threshold) return x + threshold;
        return 0.0;
    }

    /** Apply soft threshold to entire vector */
    public static double[] softThreshold(double[] x, double threshold) {
        double[] r = new double[x.length];
        for (int i = 0; i < x.length; i++) r[i] = softThreshold(x[i], threshold);
        return r;
    }
}
