package com.weather.prophet.compute;

/**
 * Pure Java CPU compute backend. Always available as fallback.
 */
public class CPUBackend implements ComputeBackend {

    @Override
    public Type getType() { return Type.CPU; }

    @Override
    public double[][] matmul(double[][] A, double[][] B) {
        int m = A.length, n = A[0].length, p = B[0].length;
        double[][] C = new double[m][p];
        for (int i = 0; i < m; i++)
            for (int k = 0; k < n; k++) {
                double a = A[i][k];
                if (a == 0) continue;
                for (int j = 0; j < p; j++)
                    C[i][j] += a * B[k][j];
            }
        return C;
    }

    @Override
    public double[] vecAdd(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] + b[i];
        return r;
    }

    @Override
    public double[] vecMul(double[] a, double[] b) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] * b[i];
        return r;
    }

    @Override
    public double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    @Override
    public double[] vecScale(double[] a, double s) {
        double[] r = new double[a.length];
        for (int i = 0; i < a.length; i++) r[i] = a[i] * s;
        return r;
    }

    @Override
    public double[][] computeFourierFeatures(double[] t, int fourierOrder, double period) {
        int n = t.length;
        int numFeatures = 2 * fourierOrder;
        double[][] X = new double[n][numFeatures];
        for (int i = 0; i < n; i++) {
            for (int k = 1; k <= fourierOrder; k++) {
                double angle = 2.0 * Math.PI * k * t[i] / period;
                X[i][2 * (k - 1)] = Math.cos(angle);
                X[i][2 * (k - 1) + 1] = Math.sin(angle);
            }
        }
        return X;
    }

    @Override
    public boolean isAvailable() { return true; }

    @Override
    public void release() { /* no-op */ }
}
