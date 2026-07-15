package com.weather.prophet.matrix;

/**
 * Lightweight matrix operations library for Prophet model.
 * Implements basic linear algebra: multiply, transpose, solve (via Cholesky/LU).
 * No external dependencies required.
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

    public int getRows() { return rows; }
    public int getCols() { return cols; }

    public double get(int i, int j) { return data[i][j]; }
    public void set(int i, int j, double val) { data[i][j] = val; }

    /**
     * Matrix multiplication: this * other
     */
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                "Dimension mismatch: (" + this.rows + "x" + this.cols +
                ") * (" + other.rows + "x" + other.cols + ")");
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int k = 0; k < this.cols; k++) {
                double aik = this.data[i][k];
                if (aik == 0.0) continue;
                for (int j = 0; j < other.cols; j++) {
                    result.data[i][j] += aik * other.data[k][j];
                }
            }
        }
        return result;
    }

    /**
     * Matrix transpose
     */
    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[j][i] = data[i][j];
            }
        }
        return result;
    }

    /**
     * Solve the linear system A * x = b using Gaussian elimination with partial pivoting.
     * A is this matrix (must be square), b is a column vector.
     * Returns x as a 1D array.
     */
    public double[] solve(double[] b) {
        if (rows != cols) throw new IllegalArgumentException("Matrix must be square");
        if (b.length != rows) throw new IllegalArgumentException("Dimension mismatch");

        int n = rows;
        // Augmented matrix [A | b]
        double[][] aug = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[i], 0, aug[i], 0, n);
            aug[i][n] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < n; col++) {
            // Find pivot
            int maxRow = col;
            double maxVal = Math.abs(aug[col][col]);
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(aug[row][col]) > maxVal) {
                    maxVal = Math.abs(aug[row][col]);
                    maxRow = row;
                }
            }
            // Swap rows
            double[] temp = aug[col];
            aug[col] = aug[maxRow];
            aug[maxRow] = temp;

            if (Math.abs(aug[col][col]) < 1e-12) {
                // Add small regularization for near-singular matrices
                aug[col][col] = 1e-8;
            }

            // Eliminate below
            for (int row = col + 1; row < n; row++) {
                double factor = aug[row][col] / aug[col][col];
                for (int j = col; j <= n; j++) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            x[i] = aug[i][n];
            for (int j = i + 1; j < n; j++) {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }
        return x;
    }

    /**
     * Solve the normal equations: (X^T * X) * beta = X^T * y
     * This is the core of least-squares fitting used in Prophet.
     * Uses ridge regularization (L2) for numerical stability.
     *
     * @param X the design matrix
     * @param y the target vector
     * @param lambda ridge regularization parameter
     * @return beta coefficients
     */
    public static double[] solveLeastSquares(Matrix X, double[] y, double lambda) {
        Matrix Xt = X.transpose();
        Matrix XtX = Xt.multiply(X);

        // Add ridge regularization to diagonal for stability
        for (int i = 0; i < XtX.rows; i++) {
            XtX.set(i, i, XtX.get(i, i) + lambda);
        }

        // Compute X^T * y  (result is p x 1, where p = number of features)
        double[] Xty = new double[Xt.rows];
        for (int i = 0; i < Xt.rows; i++) {
            for (int j = 0; j < Xt.cols; j++) {
                Xty[i] += Xt.get(i, j) * y[j];
            }
        }

        return XtX.solve(Xty);
    }

    /**
     * Create an identity matrix
     */
    public static Matrix identity(int n) {
        Matrix I = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            I.set(i, i, 1.0);
        }
        return I;
    }

    /**
     * Create a column vector from a 1D array
     */
    public static Matrix columnVector(double[] arr) {
        Matrix m = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            m.set(i, 0, arr[i]);
        }
        return m;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Matrix(%dx%d):\n", rows, cols));
        for (int i = 0; i < Math.min(rows, 5); i++) {
            sb.append("[");
            for (int j = 0; j < Math.min(cols, 8); j++) {
                sb.append(String.format("%10.4f ", data[i][j]));
            }
            if (cols > 8) sb.append("...");
            sb.append("]\n");
        }
        if (rows > 5) sb.append("...\n");
        return sb.toString();
    }
}
