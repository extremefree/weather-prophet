# Weather Prophet ☀️🌧️

A Java implementation of weather forecasting model inspired by [Facebook Prophet](https://github.com/facebook/prophet).

## Core Model

The model uses additive decomposition:

```
y(t) = g(t) + s(t) + h(t) + ε(t)
```

| Component | Method | Description |
|-----------|--------|-------------|
| **g(t)** Trend | Piecewise linear with auto changepoints | Captures long-term trend and rate changes |
| **s(t)** Seasonality | Fourier series | Models yearly and weekly periodic patterns |
| **h(t)** Holidays | Gaussian-decay window effects | Accounts for special events and holidays |
| **ε(t)** Error | Normal distribution | Provides confidence intervals |

## Project Structure

```
src/com/weather/prophet/
├── matrix/Matrix.java              # Linear algebra (multiply, transpose, Gaussian elimination, ridge LS)
├── model/
│   ├── TrendComponent.java         # Piecewise linear trend with changepoints
│   ├── SeasonalityComponent.java   # Fourier series seasonality
│   ├── HolidayComponent.java       # Holiday / special event effects
│   └── ProphetModel.java           # Main model (combines all components + CI)
├── data/
│   ├── DataPoint.java              # Data point record
│   └── WeatherDataGenerator.java   # Synthetic weather data generator
└── WeatherProphetApp.java          # Entry point (train + forecast + visualize)
```

## Quick Start

```bash
# Compile
find src -name "*.java" | xargs javac -d out

# Run
java -cp out com.weather.prophet.WeatherProphetApp
```

**No external dependencies** — all linear algebra and optimization implemented from scratch.

## Example Output

```
[ProphetModel] Training metrics - MAE: 2.33°C, RMSE: 2.91°C, R²: 0.90

  Day      | Predicted  | 80% CI         | Trend
  Day 365  |    5.67°C  | [  1.94,   9.41] |    4.13°C
  Day 366  |    6.02°C  | [  2.29,   9.76] |    4.03°C
  ...
```

## How It Works

1. **Trend Fitting**: Places 25 changepoints uniformly in the first 80% of the time range, fits a piecewise linear function via ridge-regularized least squares.

2. **Seasonality Fitting**: Uses Fourier basis functions (cos/sin pairs) to model periodic patterns. Yearly seasonality uses 10 harmonics; weekly uses 3.

3. **Holiday Fitting**: Models each holiday as a Gaussian-decayed effect within a configurable window, estimated from residuals after removing trend and seasonality.

4. **Forecasting**: Combines all components and adds confidence intervals based on residual standard error.
