# Weather Prophet

A Java implementation of weather forecasting model faithfully following [Facebook Prophet](https://github.com/facebook/prophet), including L-BFGS optimization, Bayesian priors, MCMC/NUTS sampling, and GPU (OpenCL) acceleration.

## Core Model

Additive decomposition (same as Prophet):

```
y(t) = g(t) + s(t) + h(t) + ε(t)
```

| Component | Prophet Algorithm | Java Implementation |
|-----------|-------------------|---------------------|
| **g(t)** Trend | Piecewise linear / Logistic growth with Laplace prior | `ProphetModel` + `BayesianPriors` |
| **s(t)** Seasonality | Fourier series with Normal prior | `ProphetModel` (Fourier features) |
| **h(t)** Holidays | Gaussian-window effects with Normal prior | `ProphetModel` (holiday features) |
| **ε(t)** Error | Half-Cauchy prior on sigma | `BayesianPriors.logHalfCauchy` |
| **MAP estimate** | Stan `optimizing()` → L-BFGS | `LBFGSOptimizer` |
| **Posterior** | Stan `sampling()` → NUTS | `MCMCSampler` (NUTS + MH) |
| **GPU** | PyStan OpenCL backend | `GPUBackend` (JOCL/OpenCL) |

## Prophet Faithfulness Checklist

- [x] **Trend formula**: `g(t) = (k + A(t)^T * delta) * t + (m + A(t)^T * gamma)`
- [x] **Continuity constraint**: `gamma_j = -s_j * delta_j` (exact Prophet formula)
- [x] **Logistic growth**: `g(t) = C / (1 + exp(-(k+A^T*d) * (t - (m+A^T*gamma))))`
- [x] **Laplace prior on delta**: `delta ~ Laplace(0, tau)` → L1 sparse changepoints
- [x] **Normal prior on k, m**: `k, m ~ Normal(0, 5)` (same as Prophet Stan model)
- [x] **Half-Cauchy prior on sigma**: `sigma ~ HalfCauchy(0, scale)` (same as Prophet Stan model)
- [x] **Normal prior on beta, kappa**: `beta ~ Normal(0, sigma_beta)` (same as Prophet)
- [x] **L-BFGS optimization**: Same algorithm as Stan's `optimizing()`
- [x] **NUTS sampler**: No-U-Turn Sampler with leapfrog integration (same as Stan's `sampling()`)
- [x] **Dual-averaging step size adaptation**: Like Stan's warmup
- [x] **Posterior predictive simulation**: For uncertainty intervals
- [x] **GPU acceleration**: OpenCL kernels for matmul, Fourier features, MCMC proposals
- [x] **Data scaling**: y standardized internally (Prophet's internal scaling)
- [x] **Fourier seasonality**: Additive and multiplicative modes
- [x] **Multiplicative seasonality**: `y(t) = g(t) * (1 + s(t) + h(t))`
- [x] **Automatic changepoint placement**: First 80% of history (configurable)

## Project Structure

```
src/com/weather/prophet/
├── core/
│   ├── ProphetConfig.java          # Model configuration (priors, MCMC settings, holidays)
│   └── ProphetModel.java           # Main model: fit() + predict() + uncertainty
├── optimize/
│   ├── LBFGSOptimizer.java         # L-BFGS quasi-Newton optimizer (same as Stan)
│   └── BayesianPriors.java         # Neg log posterior + analytical gradient
├── mcmc/
│   └── MCMCSampler.java            # NUTS (No-U-Turn Sampler) + Metropolis-Hastings
├── compute/
│   ├── ComputeBackend.java         # Compute interface (CPU/GPU abstraction)
│   ├── CPUBackend.java             # CPU implementation (matrix ops, Fourier)
│   └── GPUBackend.java             # GPU implementation (OpenCL via JOCL)
├── matrix/
│   ├── Matrix.java                 # Matrix operations (multiply, solve, Cholesky)
│   └── VecOps.java                 # Vector operations (dot, add, scale, copy)
├── data/
│   └── DataPoint.java              # Data point with timestamp + value
└── WeatherProphetApp.java          # Entry point (train + forecast + visualize)
```

## Quick Start

```bash
# Compile (requires Java 17+)
find src -name "*.java" | xargs javac -d out -cp "lib/jocl-2.0.4.jar"

# Run (CPU mode - zero external dependencies at runtime)
java -cp "out:lib/jocl-2.0.4.jar" com.weather.prophet.WeatherProphetApp

# Run with GPU acceleration (requires OpenCL drivers + JOCL native library)
java -cp "out:lib/jocl-2.0.4.jar" -Djava.library.path=/usr/lib/jni \
     com.weather.prophet.WeatherProphetApp
```

## Example Output

```
========================================================================
  Weather Prophet — 天气预测模型 (Java Prophet Implementation)
  Model: y(t) = g(t) + s(t) + h(t) + ε(t)
  Trend: Piecewise linear with Laplace prior (L1 sparse changepoints)
  Optimization: L-BFGS (same as Stan's optimizing())
  Uncertainty: Posterior predictive simulation
  GPU: OpenCL acceleration via JOCL
========================================================================

[1] Generating synthetic weather data...
    365 days of temperature data generated
    Range: -3.8°C ~ 37.0°C

[2] Configuring Prophet model...
    Growth: LINEAR, Changepoints: 25 (Laplace τ=0.050)
    Seasonality: yearly(order=10) + weekly(order=3)
    MCMC: 100 samples, 50 warmup, 4 chains, NUTS=true

[3] Fitting Prophet model...
    [Prophet] L-BFGS converged: negLogPost=2659.883492
    [Prophet] Fitted: k=0.003251, m=-0.960687, sigma_obs=0.099981
    [MCMC] 4 chains complete: 400 posterior samples

[8] Model Evaluation (In-Sample):
    MAE:              3.05°C
    RMSE:             3.83°C
    R²:               0.829
    80% CI Coverage:  88.5%

[9] Algorithm Summary (Prophet Faithfulness Check):
  ✓ Model: y(t) = g(t) + s(t) + h(t) + ε(t)
  ✓ Trend: Piecewise linear g(t) = (k+A^T*δ)*t + (m+A^T*γ)
  ✓ Continuity: γ_j = -s_j * δ_j (exact Prophet formula)
  ✓ Prior on δ: Laplace(0, τ) → L1 sparse changepoints
  ✓ Prior on k,m: Normal(0, 5) (same as Prophet Stan model)
  ✓ Prior on σ: HalfCauchy(0, 0.5) (same as Prophet Stan model)
  ✓ Optimization: L-BFGS (same as Stan's optimizing())
  ✓ MCMC: NUTS (No-U-Turn Sampler, same as Stan's sampling())
  ✓ GPU: OpenCL kernels for matmul, Fourier, MCMC sampling
  ✓ Seasonality: Fourier series with Normal prior on β
  ✓ Holidays: Gaussian-window features with Normal prior on κ
```

## GPU Acceleration

The GPU backend uses [JOCL](https://github.com/gpu/JOCL) (Java bindings for OpenCL) to accelerate:

1. **Matrix multiplication** — X^T X for normal equations
2. **Fourier feature computation** — Embarrassingly parallel cos/sin
3. **Trend computation** — Piecewise linear with changepoint A matrix
4. **Logistic trend** — Saturating growth model
5. **MCMC proposals** — Parallel Metropolis-Hastings proposals
6. **Density kernels** — Laplace, Normal log-PDF (for MCMC accept/reject)

When OpenCL is not available, the model automatically falls back to the CPU backend.

## Dependencies

| Library | Purpose | Required at Runtime |
|---------|---------|-------------------|
| Java 17+ | Language runtime | Yes |
| JOCL 2.0.4 | OpenCL GPU bindings | Only for GPU mode |

CPU-only mode has **zero external dependencies**.

## License

MIT
