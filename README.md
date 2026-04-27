# whale-tracker

A CPU-only AutoResearch project for **STAT 390**.

---

## Problem

Categorize and track top performing whales on Polymarket.
**Metrics**: Brier Skill Score (higher is better)
**Data**: Polymarket dataset found on Kaggle

## Project Structure

```
demo_autoresearch/
├── prepare.py      # FROZEN — data loading, evaluation metric, plotting
├── model.py        # EDITABLE — agent modifies only this file
├── run.py          # Run a single experiment and log result
├── program.md      # Agent instructions (the agent reads this)
├── results.tsv     # Experiment log (auto-generated)
└── performance.png # Performance plot (auto-generated)
```

**Key rule**: the agent may only modify `model.py`. Everything else is frozen.

---

## Setup

### 1. Install an AI coding agent (CLI)

You need a CLI coding agent that can read files, edit files, and run shell commands.
Two recommended options — **neither requires an API key**.

#### Option A: Claude Code CLI (recommended)

```bash
# macOS / Linux / WSL — one-line install
curl -fsSL https://claude.ai/install.sh | bash

# macOS — or via Homebrew
brew install --cask claude-code

# Windows PowerShell
irm https://claude.ai/install.ps1 | iex

# Windows — or via WinGet
winget install Anthropic.ClaudeCode
```

Then launch:

```bash
cd demo_autoresearch
claude
```

First launch opens a browser for login — **no API key needed**.
Works with any Claude subscription (Pro $20/mo, Max, or Team).

Docs: https://code.claude.com/docs

#### Option B: OpenAI Codex CLI

```bash
# Install
npm install -g @openai/codex

# Launch
cd demo_autoresearch
codex
```

First launch opens a browser for ChatGPT login — **no API key needed**.
Works with ChatGPT Plus or higher.

Docs: https://github.com/openai/codex

### 2. Install Python environment

This project requires **Python 3.10+** with `scikit-learn`, `matplotlib`, and `numpy`.

#### Check if Python is installed

```bash
python3 --version
# Should print Python 3.10.x or higher
# If not installed, see below
```

#### Install Python (if needed)

```bash
# macOS
brew install python@3.12

# Ubuntu / Debian
sudo apt update && sudo apt install python3 python3-pip python3-venv

# Windows — download from https://www.python.org/downloads/
# During install, check "Add Python to PATH"
```

#### Install dependencies

```bash
# Option A: with pip (simplest)
pip install scikit-learn matplotlib numpy

# Option B: with uv (faster, used in the main autoresearch project)
# Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install scikit-learn matplotlib numpy

# Option C: with conda
conda install scikit-learn matplotlib numpy
```

#### What gets installed

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | >= 1.3 | ML models, pipelines, evaluation |
| matplotlib | >= 3.7 | Performance plotting |
| numpy | >= 1.24 | Array operations (scikit-learn dependency) |

No GPU, no PyTorch, no heavy downloads — everything runs on CPU.

### 3. Verify setup

```bash
# Quick check: all imports work
python3 -c "import sklearn, matplotlib, numpy; print('All good')"

# Full check: run one experiment
python3 run.py "test run"
# Expected output:
#   Data: 16512 train, 4128 val, 8 features
#   val_rmse: 0.745581
#   val_r2:   0.575788
#   Result logged to results.tsv

# Clean up test result
rm -f results.tsv
```

---

## How to Run the Agent Loop

### Quick start (copy-paste this prompt into your agent)

```
Read program.md for your instructions, then read model.py.
Run `python run.py "baseline"` to establish the baseline RMSE.
Then enter the AutoResearch loop:

1. Propose one modification to model.py (e.g., different estimator,
   feature engineering, hyperparameter change).
2. Edit model.py with your change.
3. Run: python run.py "<short description of what you changed>"
4. Compare the new val_rmse to the current best.
   - If improved: KEEP the change, note the new best.
   - If worse: REVERT model.py to the previous version.
5. Repeat from step 1. Try at least 6 different ideas.

After all iterations, run `python prepare.py` to generate performance.png.
Print a summary table of all experiments and which were kept vs discarded.
```

### More specific prompt (if you want to control the search)

```
You are an AutoResearch agent. Read program.md for rules.

Your job: maximize BSS on master_trades.parquet by modifying model.py.

Constraints:
- model.py must define build_model() returning an sklearn estimator
- Do NOT modify prepare.py or run.py
- Each experiment must finish in < 90 seconds
- Do NOT remove core features: whale_flow_top10p_quote, hhi, top10_share

Search strategy:
1. Start with baseline (LogisticRegression using only p_close)
2. Add core whale features to measure the informational edge (Whale Baseline)
3. Try tree ensembles (RandomForest, HistGradientBoosting) to capture non-linear whale interactions
4. Try feature exploration by adding context features (atr_14, gini, vol_quote) to the whale core
5. Try hyperparameter tuning on the best model so far

For each experiment:
- Run: python run.py "<description>"
- If BSS improved → keep
- If BSS worsened → revert model.py to previous version
- Log your reasoning for each decision

After finishing, run: python prepare.py
```

---

## Plotting Results

After running experiments:

```bash
python prepare.py
# Generates performance.png from results.tsv
```

This produces a two-panel chart:
- Top: validation Brier Skill Score (BSS) over iterations (green = keep, red = discard, blue = baseline)
- Bottom: validation Brier Score (BS) over iterations
- Green line: best-so-far Brier Skill Score (BSS) envelope

---
