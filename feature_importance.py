"""
Feature Importance — Monte Carlo Simulation
--------------------------------------------
For each variable:
  1. Fit logistic regression → get P(injury) for every athlete
  2. Run 10,000 MC trials: randomly draw injury outcomes from those probabilities
  3. Look at the distribution of simulated injury rates

Core idea: a STRONG predictor produces P(injury) values that are confident
(close to 0 or 1). Bernoulli draws from confident probabilities produce tight,
stable simulations. A WEAK predictor gives P ≈ 0.5 for everyone — maximum
noise, wide simulated distribution.

Ranking metric: CI width — narrower = more confident = better predictor.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# ── Setup ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/highAccuracySportInjuryDataset.csv').dropna()
rng = np.random.default_rng(42)
N_RUNS = 10_000

y = df['Injury_Risk'].values
TRUE_RATE = y.mean()
FEATURES = [c for c in df.columns if c != 'Injury_Risk']

# ── Monte Carlo per feature ────────────────────────────────────────────────────
results = {}

for feat in FEATURES:
    X = df[[feat]].values
    model = LogisticRegression(C=1e6, random_state=0)
    model.fit(X, y)
    p_hat = model.predict_proba(X)[:, 1]

    # 10,000 trials: each athlete gets a random Bernoulli draw from their p_hat
    mc = rng.binomial(1, p_hat, size=(N_RUNS, len(y))).mean(axis=1)

    lo, hi = np.percentile(mc, [2.5, 97.5])
    results[feat] = dict(
        mean=mc.mean(),
        lo=lo, hi=hi,
        width=.1 - (hi - lo),
        mc=mc,
    )

# Rank: narrowest CI = most confident predictions = best predictor
ranked = sorted(results.items(), key=lambda kv: kv[1]['width'])

# ── Print table ────────────────────────────────────────────────────────────────
print(f"\nFeature Importance — Monte Carlo  ({N_RUNS:,} trials per feature)")
print(f"True injury rate: {TRUE_RATE:.1%}  —  narrower CI = stronger predictor\n")
print(f"  {'#':<4} {'Feature':<22} {'MC Mean':>8}  {'95% CI':>16}  {'CI Width':>9}  Confidence")
print(f"  {'-'*75}")

min_w = min(v['width'] for v in results.values())
max_w = max(v['width'] for v in results.values())

for rank, (feat, r) in enumerate(ranked, 1):
    strength = 1 - (r['width'] - min_w) / (max_w - min_w + 1e-9)
    bar = '█' * int(strength * 20)
    label = 'Strong' if strength > 0.66 else 'Moderate' if strength > 0.33 else 'Weak'
    print(f"  {rank:<4} {feat:<22} {r['mean']:>8.1%}  [{r['lo']:.1%}, {r['hi']:.1%}]  {r['width']:>9.4f}  {bar} {label}")

# ── Plot ───────────────────────────────────────────────────────────────────────
feats  = [f for f, _ in ranked]
widths = [results[f]['width'] for f in feats]

# color by CI width: green = tight (strong), red = wide (weak)
norm = plt.Normalize(min(widths), max(widths))
cmap = plt.cm.RdYlGn_r
colors = [cmap(norm(w)) for w in widths]

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(
    f'Which Variable Best Predicts Injury?  —  Monte Carlo ({N_RUNS:,} trials each)\n'
    f'Narrower simulated CI = more confident predictions = stronger predictor',
    fontsize=12, fontweight='bold'
)

ax.bar(feats, widths, color=colors, edgecolor='white', width=0.65)

ax.set_ylabel('95% CI Width of Simulated Injury Rate  (narrower = better)', fontsize=10)
ax.set_title('Ranking by Prediction Confidence', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right', fontsize=8)

plt.tight_layout()
out = Path('results/figures')
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / 'feature_importance.png', dpi=300, bbox_inches='tight')
print(f'\nFigure saved → results/figures/feature_importance.png')
plt.show()