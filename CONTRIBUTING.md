# Contributing to NSE-Neuron 🤝

<div align="center">
  <img src="images/nse-neuron-logo.svg" alt="NSE-Neuron Logo" width="350"/>
  <p><strong>Thank you for your interest in contributing to NSE-Neuron!</strong></p>
  <p>Whether you're fixing a bug, improving a model, adding a new feature, or improving documentation — every contribution matters. 🙌</p>
</div>

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [How Can I Contribute?](#-how-can-i-contribute)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Development Guidelines](#-development-guidelines)
- [Submitting a Pull Request](#-submitting-a-pull-request)
- [Reporting Issues](#-reporting-issues)
- [Ideas & Feature Requests](#-ideas--feature-requests)
- [Community](#-community)

---

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a **respectful, inclusive, and collaborative environment**. Be kind, be constructive, and help each other grow.

- ✅ Be welcoming to newcomers
- ✅ Use inclusive language
- ✅ Accept constructive criticism gracefully
- ❌ No harassment, trolling, or personal attacks
- ❌ No spam or self-promotion in issues/PRs

---

## 💡 How Can I Contribute?

There are many ways to contribute, regardless of your skill level:

| Type | Examples |
|------|----------|
| 🐛 **Bug Fix** | Fix errors in model predictions, data fetching, plotting |
| ✨ **New Feature** | Add a new forecasting model, new stock indicators |
| 📊 **Model Improvement** | Tune hyperparameters, improve accuracy, add new architectures |
| 📖 **Documentation** | Improve README, add docstrings, write tutorials |
| 🧪 **Tests** | Write unit tests for models and utilities |
| 🎨 **UI/UX** | Improve terminal output, plot styling |
| 🔧 **Refactoring** | Clean up code, improve structure, follow OOP best practices |

---

## 🚀 Getting Started

### 1. Fork & Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/NSE-Neuron.git
cd NSE-Neuron
```

### 2. Set Up Your Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Create a Branch

Always work on a **dedicated branch** — never directly on `main`.

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### 4. Make Your Changes

Write your code, add docstrings, and test locally:

```bash
python main.py
```

### 5. Commit & Push

```bash
git add .
git commit -m "feat: add Transformer model for forecasting"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request

Go to the original repo on GitHub and open a **Pull Request** against the `main` branch. Fill in the PR template with a clear description.

---

## 📁 Project Structure

```
NSE-Neuron/
├── main.py                  # Entry point
├── config.py                # Hyperparameters, constants, thresholds
├── requirements.txt
├── Algorithms/              # Forecasting algorithm scripts
├── Dataset/                 # Data fetching decorator & helpers
├── models/                  # Deep learning model classes (OOP)
│   ├── base_model.py        # Abstract BaseModel
│   ├── lstm.py
│   ├── bilstm.py
│   ├── gru.py
│   ├── cnn_lstm.py
│   └── lstm_classifier.py
├── utils/
│   ├── data_fetcher.py      # NSE data fetching + CSV caching
│   └── preprocessor.py      # Data preprocessing pipeline
├── visualization/
│   └── ploting.py           # Candlestick + forecast plots
├── data/raw/                # Cached CSV files
├── saved_models/            # Saved model weights
└── images/                  # Screenshots and logo
```

---

## 🛠️ Development Guidelines

### Adding a New Model

1. Create `models/your_model.py`
2. Inherit from `BaseModel` in `base_model.py`
3. Implement required abstract methods: `build()`, `train()`, `predict()`
4. Export a top-level function (e.g. `def your_model(df):`) for use in `main.py`
5. Add it as a new `case` in the `match` block in `main.py`
6. Update the models table in `README.md`

```python
# Example skeleton
from models.base_model import BaseModel

class YourModel(BaseModel):
    def build(self): ...
    def train(self, X, y): ...
    def predict(self, X): ...

def your_model(df):
    model = YourModel(df)
    # ... train and return predictions
    return pred, rmse
```

### Config-Driven Development

All hyperparameters, thresholds, and constants **must** come from `config.py`. Do not hardcode values inside model files.

```python
# ✅ Good
from config import EPOCHS, UNITS, TIME_STEP, FORECAST_DAYS

# ❌ Avoid
epochs = 50
units = 64
```

### Code Style

- Follow **PEP 8** conventions
- Use **meaningful variable names** (avoid `x1`, `temp`, `data2`)
- Add **docstrings** to all public functions and classes
- Keep functions **focused** — one function, one responsibility
- Use `print()` sparingly; prefer structured output via `tabulate`

---

## 📬 Submitting a Pull Request

Please make sure your PR:

- [ ] Is based on the latest `main` branch
- [ ] Has a clear title and description
- [ ] References any related issue (e.g. `Closes #12`)
- [ ] Does not break existing functionality (`python main.py` runs cleanly)
- [ ] Follows the code style guidelines above
- [ ] Includes docstrings for any new functions/classes

**PR Title Format:**

```
feat: add Transformer-based forecasting model
fix: correct date alignment in candlestick plot
docs: update README with new model comparison table
refactor: move preprocessing to utils/preprocessor.py
```

---

## 🐞 Reporting Issues

Found a bug? Please open a [GitHub Issue](https://github.com/NayakwadiS/NSE-Neuron/issues) with:

1. **Description** — What happened vs. what you expected
2. **Steps to Reproduce** — Exact commands / inputs used
3. **Error Output** — Full traceback if applicable
4. **Environment** — Python version, OS, dependency versions (`pip freeze`)

```
Python: 3.10.x
OS: Windows 11 / Ubuntu 22.04
TensorFlow: 2.x.x
nselib: x.x.x
```

---

## 🌟 Ideas & Feature Requests

Have an idea to make NSE-Neuron better? Open a [GitHub Discussion](https://github.com/NayakwadiS/NSE-Neuron/discussions) or create an Issue with the label `enhancement`.

Some areas where contributions are especially welcome:

- 🔮 **New Models** — Transformer, SARIMA, Prophet, XGBoost
- 📊 **Technical Indicators** — RSI, MACD, Bollinger Bands as input features
- 🌐 **Web UI** — Flask / Streamlit dashboard
- 📱 **Alerts** — Email/Telegram notification for BUY/SELL signals
- 🗂️ **Multi-stock** — Batch forecasting for a portfolio
- 🧪 **Backtesting** — Validate predictions against historical performance
- 📦 **Packaging** — Turn NSE-Neuron into an installable PyPI package

---

## 🏆 Community

- ⭐ **Star the repo** if you find it useful — it helps others discover the project!
- 🍴 **Fork it** and experiment freely
- 💬 Open a **Discussion** for questions or ideas
- 🔗 Share your results with the community

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sujitnayakwadi/)

---

<div align="center">
  <p>Built with ❤️ for the Indian Stock Market community</p>
  <p><em>Happy forecasting — and happy contributing! 🚀</em></p>
</div>

