package com.weather.prophet.matrix;

/**
 * High-performance matrix operations library.
 * Supports both CPU and GPU (via ComputeBackend) computation.
 * All core linear algebra needed for Prophet's L-BFGS optimization and MCMC.
 */
public class Matrix {
    private final double[][] data;
    private final int rows;
    private final int cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, this.data[i], 0, cols);
        }
    }

    public static Matrix column(double[] arr) {
        Matrix m = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) m.set(i, 0, arr[i]);
        return m;
    }

    public static Matrix zeros(int r, int c) { return new Matrix(r, c); }
    public static Matrix identity(int n) {
        Matrix I = new Matrix(n, n);
        for (int i = 0; i < n; i++) I.set(i, i, 1.0);
        return I;
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }
    public double get(int i, int j) { return data[i][j]; }
    public void set(int i, int j, double v) { data[i][j] = v; }

    public double[] getColumn(int j) {
        double[] col = new double[rows];
        for (int i = 0; i < rows; i++) col[i] = data[i][j];
        return col;
    }

    public double[] getRow(int i) {
        return data[i].clone();
    }

    public double[] flatten() {
        double[] arr = new double[rows * cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                arr[i * cols + j] = data[i][j];
        return arr;
    }

    public Matrix transpose() {
        Matrix r = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[j][i] = data[i][j];
        return r;
    }

    public Matrix multiply(Matrix o) {
        if (cols != o.rows) throw new IllegalArgumentException("Dimension mismatch");
        Matrix r = new Matrix(rows, o.cols);
        for (int i = 0; i < rows; i++)
            for (int k = 0; k < cols; k++) {
                double a = data[i][k];
                if (a == 0) continue;
                for (int j = 0; j < o.cols; j++)
                    r.data[i][j] += a * o.data[k][j];
            }
        return r;
    }

    public Matrix add(Matrix o) {
        Matrix r = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[i][j] = data[i][j] + o.data[i][j];
        return r;
    }

    public Matrix subtract(Matrix o) {
        Matrix r = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[i][j] = data[i][j] - o.data[i][j];
        return r;
    }

    public Matrix scale(double s) {
        Matrix r = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[i][j] = data[i][j] * s;
        return r;
    }

    /** Element-wise multiply */
    public Matrix elemMultiply(Matrix o) {
        Matrix r = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                r.data[i][j] = data[i][j] * o.data[i][j];
        return r;
    }

    /** Solve A*x = b via Gaussian elimination with partial pivoting */
    public double[] solve(double[] b) {
        if (rows != cols) throw new IllegalArgumentException("Matrix must be square");
        int n = rows;
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, aug[i], 0, n);
            aug[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int maxRow = col;
            double maxVal = Math.abs(aug[col][col]);
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > maxVal) { maxVal = Math.abs(aug[row][col]); maxRow = row; }
            }
            double[] tmp = aug[col]; aug[col] = aug[maxRow]; aug[maxRow] = tmp;
            if (Math.abs(aug[col][col]) < 1e-12) aug[col][col] = 1e-8;
            for (int row = col + 1; row < n; row++) {
                double f = aug[row][col] / aug[col][col];
                for (int j = col; j <= n; j++) aug[row][j] -= f * aug[col][j];
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) x[i] -= aug[i][j] * x[j];
            x[i] /= aug[i][i];
        }
        return x;
    }

    /** Ridge-regularized least squares: (X^TX + λI)β = X^Ty */
    public static double[] solveLeastSquares(Matrix X, double[] y, double lambda) {
        Matrix Xt = X.transpose();
        Matrix XtX = Xt.multiply(X);
        for (int i = 0; i < XtX.rows; i++) XtX.set(i, i, XtX.get(i, i) + lambda);
        double[] Xty = new double[Xt.rows];
        for (int i = 0; i < Xt.rows; i++)
            for (int j = 0; j < Xt.cols; j++)
                Xty[i] += Xt.get(i, j) * y[j];
        return XtX.solve(Xty);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Matrix(%dx%d)", rows, cols));
        return sb.toString();
    }
}
