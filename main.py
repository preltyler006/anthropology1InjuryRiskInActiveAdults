"""
Monte Carlo Injury Simulation — 3 Predictors
---------------------------------------------
For each predictor (Training_Intensity, Sleep_Hours, Injury_History):
  1. Scatter of predictor vs Injury_Risk with fitted logistic curve
  2. 10,000 MC runs drawing Bernoulli outcomes from the fitted probabilities
  3. Distribution of simulated injury rates
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
df  = pd.read_csv('data/highAccuracySportInjuryDataset.csv').dropna()
rng = np.random.default_rng(42)
N_RUNS = 10_000

PREDICTORS = [
    ('Training_Intensity', 'Training Intensity (1–10)', '#e74c3c'),
    ('Sleep_Hours',        'Sleep Hours per Night',     '#2ecc71'),
    ('Injury_History',     'Prior Injury History (0/1)', '#3498db'),
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def logistic(x, a, b):
    return expit(a * x + b)

def run_analysis(col, y):
    x = df[col].values
    lr = LogisticRegression(C=1e6, random_state=42)
    lr.fit(x.reshape(-1, 1), y)
    a, b  = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    p_hat = logistic(x, a, b)
    mc    = np.mean(rng.binomial(1, p_hat, size=(N_RUNS, len(p_hat))), axis=1)
    return x, a, b, p_hat, mc

# ── Run all three ─────────────────────────────────────────────────────────────
y = df['Injury_Risk'].values

results = []
for col, label, color in PREDICTORS:
    x, a, b, p_hat, mc = run_analysis(col, y)
    results.append((col, label, color, x, a, b, p_hat, mc))
    sign = '+' if b >= 0 else ''
    print(f"{col:<22}  P(injury) = σ({a:+.3f}·x {sign}{b:.3f})"
          f"   MC mean {mc.mean():.1%} ± {mc.std():.1%}"
          f"   95% CI [{np.percentile(mc,2.5):.1%}, {np.percentile(mc,97.5):.1%}]")

print(f"\nEmpirical injury rate: {y.mean():.1%}")

# ── Plot: 3 rows × 2 columns ──────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(13, 14))
fig.suptitle('Monte Carlo Injury Simulation  (N = 10,000 runs, n = 600 athletes)',
             fontsize=14, fontweight='bold', y=1.01)

for row, (col, label, color, x, a, b, p_hat, mc) in enumerate(results):
    ax_fit, ax_mc = axes[row]

    # ── Left: scatter + logistic curve ───────────────────────────────────────
    jitter = rng.uniform(-0.02, 0.02, size=len(y))
    ax_fit.scatter(x, y + jitter, alpha=0.22, s=12, color=color, label='Observed (jittered)')

    x_line = np.linspace(x.min(), x.max(), 300)
    sign = '+' if b >= 0 else ''
    ax_fit.plot(x_line, logistic(x_line, a, b), color='black', linewidth=2.2,
                label=f'σ({a:+.2f}·x {sign}{b:.2f})')

    ax_fit.set_xlabel(label)
    ax_fit.set_ylabel('P(Injury)')
    ax_fit.set_title(f'Logistic Fit — {col}')
    ax_fit.set_ylim(-0.08, 1.08)
    ax_fit.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_fit.legend(fontsize=9)
    ax_fit.grid(alpha=0.3)

    # ── Right: MC distribution ────────────────────────────────────────────────
    ax_mc.hist(mc, bins=60, color=color, edgecolor='white', alpha=0.8, density=True)
    ax_mc.axvline(y.mean(),       color='black',  lw=2,   ls='--',
                  label=f'Observed  {y.mean():.1%}')
    ax_mc.axvline(mc.mean(),      color='orange', lw=2,   ls='-',
                  label=f'MC mean  {mc.mean():.1%}')
    ax_mc.axvline(np.percentile(mc, 2.5),  color='gray', lw=1.2, ls=':', label='95% CI')
    ax_mc.axvline(np.percentile(mc, 97.5), color='gray', lw=1.2, ls=':')

    ax_mc.set_xlabel('Injury Rate (fraction injured)')
    ax_mc.set_ylabel('Density')
    ax_mc.set_title(f'MC Distribution — {col}')
    ax_mc.legend(fontsize=9)
    ax_mc.grid(alpha=0.3)

plt.tight_layout()
out = Path('results/figures')
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'mc_simulation.png', dpi=300, bbox_inches='tight')
print(f'\nFigure saved → results/figures/mc_simulation.png')
plt.show()
