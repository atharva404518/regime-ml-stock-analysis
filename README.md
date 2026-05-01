# Regime-Sensitive Performance Analysis of Machine Learning Models for Stock Prediction

A CLI-based quantitative research framework designed to evaluate how different
machine learning models perform across varying market regimes. The project
emphasizes robustness, stability, and real-world trading performance rather than
just predictive accuracy.

---

## Overview

This project analyzes stock return prediction using multiple machine learning
models and evaluates their behavior under different market conditions such as
trend and volatility regimes.

The framework integrates:

- Data ingestion from external APIs
- Feature engineering for financial time series
- Regime classification (trend and volatility)
- Walk-forward validation
- Trading strategy simulation
- Performance evaluation using both statistical and financial metrics

---

## Models Implemented

- Linear Regression
- Ridge Regression
- Random Forest
- Long Short-Term Memory (LSTM)

---

## Key Features

- Regime-aware performance analysis
- Walk-forward (time-series safe) validation
- Backtesting with transaction costs
- Long/Short trading simulation
- Composite scoring and model ranking
- Experiment tracking and result storage

---

## Market Regime Framework

Market conditions are classified based on:

- **Trend Regime**\
  Derived using moving average relationships (e.g., short-term vs long-term)

- **Volatility Regime**\
  Based on rolling standard deviation of returns

- **Crash Detection**\
  Extreme negative returns flagged as crash events

---

## Evaluation Metrics

Models are evaluated using both prediction and trading metrics:

- RMSE (Root Mean Squared Error)
- Directional Accuracy
- Sharpe Ratio
- Information Coefficient (IC)
- Cumulative Returns
- Regime Dispersion

---

## Project Structure

project/ │ ├── main.py ├── config.py ├── requirements.txt ├── README.md │ ├──
data/ │ ├── loader.py │ ├── sources/ │ ├── datasets/ │ ├── features/ ├──
regimes/ ├── models/ ├── evaluation/ ├── backtesting/ ├── experiments/ ├──
utils/

---

## Installation

Install dependencies:

````bash
pip install -r requirements.txt

## Usage

Run the CLI with default full analysis:

```bash
python main.py \
  --ticker SPY \
  --start 2000-01-01 \
  --end 2026-01-01 \
  --model all \
  --data_mode tiingo \
  --save_results \
  --analysis_mode auto \
  --advanced_analysis
````

---

## Key CLI Arguments

| Argument              | Description                                                |
| --------------------- | ---------------------------------------------------------- |
| `--ticker`            | Stock symbol (e.g., SPY, AAPL)                             |
| `--start`, `--end`    | Date range for analysis                                    |
| `--model`             | Model selection: `linear`, `ridge`, `rf`, `lstm`, or `all` |
| `--data_mode`         | `tiingo`, `local`, or `synthetic`                          |
| `--analysis_mode`     | `auto` or `custom`                                         |
| `--advanced_analysis` | Enables extended diagnostics                               |
| `--save_results`      | Saves outputs to disk                                      |

---

## Example Runs

### Run Single Model (Ridge)

```bash
python main.py --ticker SPY --model ridge --data_mode tiingo
```

### Run All Models with Full Analysis

```bash
python main.py --ticker SPY --model all --analysis_mode auto --advanced_analysis
```

### Run with Walk-Forward Validation

```bash
python main.py \
  --ticker SPY \
  --model all \
  --use_walk_forward \
  --analysis_mode auto
```

---

## Methodology Overview

### 1. Data Processing

Historical price data is collected and cleaned. Daily returns are computed and
missing values are removed.

### 2. Feature Engineering

Features include lagged returns and rolling statistics such as mean and
volatility. Only past data is used to prevent lookahead bias.

### 3. Regime Detection

Market regimes are defined using:

- Trend: Moving average relationships
- Volatility: Rolling standard deviation
- Crash detection: Extreme negative returns

### 4. Model Training

Models are trained using time-series-aware splits or walk-forward validation.

### 5. Prediction to Signal Conversion

Positive prediction leads to a long position, negative prediction leads to a
short position, and very small signals are filtered using an epsilon threshold.
Positions are shifted forward by one timestep to eliminate lookahead bias.

### 6. Backtesting

Strategies are simulated using non-overlapping holding periods, transaction
costs, and equity curve tracking.

### 7. Evaluation Metrics

- RMSE (prediction error)
- Directional Accuracy
- Sharpe Ratio (risk-adjusted return)
- Information Coefficient (IC)
- Cumulative Returns
- Regime Stability

---

## Results Interpretation

Models typically achieve approximately 55–59% directional accuracy. The
Information Coefficient indicates the presence of weak predictive signals.
However, the best Sharpe ratio (approximately 0.29) is lower than a random
benchmark (approximately 0.76), indicating limited economic viability of these
predictions.

---

## Output Files

| Path                                  | Description        |
| ------------------------------------- | ------------------ |
| `experiments/results/`                | Per-run results    |
| `experiments/summary/leaderboard.csv` | Model rankings     |
| `experiments/metadata/run_log.json`   | Run logs           |
| `data/datasets/versions/`             | Versioned datasets |

---

## Key Insights

Linear models perform competitively with more complex models. LSTM does not
guarantee better performance on financial time-series data. Predictive signals
tend to degrade when translated into trading strategies. Overall, the market
exhibits characteristics consistent with weak-form efficiency.

---

## Future Improvements

Potential improvements include incorporating macroeconomic and sentiment-based
features, exploring ensemble learning approaches, using higher-frequency data,
and refining regime detection methods.

---

## License

This project is intended for academic and research purposes only.
