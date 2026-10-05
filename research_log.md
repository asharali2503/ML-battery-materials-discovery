# Research Log

## Phase 1: Dataset Investigation

## Phase 2: Domain Generalization and Materials Project Integration (2026-10-05)
The ML-007 GroupKFold evaluation demonstrated that composition-only features suffer severely from **Chemical Domain Overfitting**. While random cross-validation yielded an $R^2$ of ~0.67, true zero-shot domain extrapolation (predicting unseen anion families) plummeted the $R^2$ to ~0.11. 

Consequently, we must restrict our subsequent Materials Project (MP) screening to the model's *known applicability domain* (e.g., specific known anions like Oxides and Sulfides) to ensure physical reliability in our conductivity predictions. We are integrating the modern `mp-api` to query thermodynamically stable, lithium-containing candidates strictly within these validated domains.

## Phase 3: Feature Explainability (2026-10-05)
To interpret the Random Forest model's physical rule learning, we attempted to generate a SHAP summary plot. Due to strict OS-level DLL application control policies blocking `shap` dependencies, the final visualization (`shap_summary.png`) relies on the model's native Gini feature importances. 

The top 3 predictive features driving solid-state ionic conductivity were found to be:
1. `MagpieData mean CovalentRadius` (~0.21 importance)
2. `MagpieData avg_dev SpaceGroupNumber` (~0.08 importance)
3. `MagpieData mean Electronegativity` (~0.05 importance)
