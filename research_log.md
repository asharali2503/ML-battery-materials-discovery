# Research Log

## Phase 1: Dataset Investigation

## Phase 2: Domain Generalization and Materials Project Integration (2026-10-05)
The ML-007 GroupKFold evaluation demonstrated that composition-only features suffer severely from **Chemical Domain Overfitting**. While random cross-validation yielded an $R^2$ of ~0.67, true zero-shot domain extrapolation (predicting unseen anion families) plummeted the $R^2$ to ~0.11. 

Consequently, we must restrict our subsequent Materials Project (MP) screening to the model's *known applicability domain* (e.g., specific known anions like Oxides and Sulfides) to ensure physical reliability in our conductivity predictions. We are integrating the modern `mp-api` to query thermodynamically stable and metastable, lithium-containing candidates (energy_above_hull <= 0.05) strictly within these validated domains.

## Phase 3: Feature Explainability (2026-10-05)
To interpret the Random Forest model's physical rule learning, we attempted to generate a SHAP summary plot. Due to strict OS-level DLL application control policies blocking `shap` dependencies, the final visualization (`shap_summary.png`) relies on the model's native Gini feature importances. 

The top 3 predictive features driving solid-state ionic conductivity were found to be:
1. `MagpieData mean CovalentRadius` (~0.21 importance)
2. `MagpieData avg_dev SpaceGroupNumber` (~0.08 importance)
3. `MagpieData mean Electronegativity` (~0.05 importance)

## Phase 4: Data Quality Audit & Pipeline Re-synchronization (2026-10-06)
Following a rigorous scientific audit, the variance threshold for duplicate compositions was tightened from 2.0 down to 1.0 orders of magnitude. This strictly removed 18 highly ambiguous polymorphic/noise entries, reducing our baseline to 500 high-confidence unique solid-state materials. 

The entire machine learning pipeline was synchronized and re-evaluated on this new baseline:
- **GroupKFold Domain Extrapolation:** The mean $R^2$ slightly improved to **0.1172** (up from 0.1103), while MAE improved to **1.9885** (down from 2.10). The core finding remains identical: composition-only models still suffer from massive Chemical Domain Overfitting and cannot safely zero-shot extrapolate out-of-domain.
- **Top MP Candidate:** The retrained Random Forest reaffirmed **$LiLa_{5}Ti_{8}O_{24}$ (LLTO)** as the most promising metastable (yet practically synthesizable) candidate in the sample batch, shifting its predicted room-temperature log10(conductivity) slightly from -3.78 to **-4.03** (~ $9.33 \times 10^{-5}$ S/cm).
