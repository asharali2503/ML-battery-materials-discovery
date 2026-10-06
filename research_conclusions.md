# Solid-State Battery Materials: Machine Learning Screening Conclusions

## Project Objective
The primary objective of this project was to predict the room-temperature ionic conductivity of lithium-ion solid electrolytes using strictly composition-based Magpie features, enabling rapid computational screening of thermodynamically stable candidates from the Materials Project.

## Methodological Rigor
To ensure physical reliability and eliminate data leakage, we implemented a rigorous data processing and validation pipeline:
- **Duplicate Aggregation:** We identified identical chemical formulas in the raw dataset with over 100x variance in measured conductivity (due to polymorphism or inter-lab variance). These unstable labels were explicitly dropped, and low-variance duplicates were aggregated via their median log-conductivity.
- **Applicability Domain Constraints:** Using a stringent `GroupKFold` validation strategy based on the primary anion, we uncovered severe **Chemical Domain Overfitting**. While random cross-validation achieved an $R^2$ of ~0.67, zero-shot extrapolation to unseen anion families dropped the $R^2$ to ~0.11. Consequently, our final candidate screening was strictly confined to known applicability domains (e.g., Oxides and Sulfides) to prevent unsafe theoretical extrapolation.

## Feature Drivers
By opening the Random Forest "black box" (via native Gini feature importances), we identified the primary compositional drivers of conductivity:
1. **`MagpieData mean CovalentRadius`:** The dominant predictor. This likely acts as a structural proxy for the average interstitial volume or bottleneck size within the crystal lattice, which physically governs lithium-ion migration.
2. **`MagpieData avg_dev SpaceGroupNumber`:** This is a composition-only model. Features such as `SpaceGroupNumber` represent composition-weighted averages of the elemental standard states, NOT the true crystallographic structure or symmetry of the compound. While mathematically predictive, this repository explicitly acknowledges that true structural information (such as distinct polymorphs or specific lattice arrangements) cannot be resolved by these elemental proxy features.
3. **`MagpieData mean Electronegativity`:** A known proxy for lattice polarizability. A softer, more polarizable anion framework (lower average electronegativity) typically weakens the electrostatic binding between the lithium ion and the lattice, facilitating higher mobility.

*Caveat:* While these features possess strong physical intuition, they represent predictive correlations rather than continuous physical laws. In composition-only models, features like electronegativity may partially act as mathematical proxies separating distinct chemical families (e.g., oxides vs. sulfides) rather than capturing continuous physical mechanisms.

## Top Candidate Identification
After training on the complete, cleaned dataset, we executed a domain-restricted query against the modern Materials Project API to retrieve thermodynamically stable, lithium-containing candidates. 

Our pipeline identified **$LiLa_{5}Ti_{8}O_{24}$** as the most promising novel solid-state electrolyte from the test batch:
- **Energy Above Hull:** 0.047 eV/atom (near thermodynamic stability)
- **Predicted log10(Conductivity):** -3.78 (~1.65 $\times 10^{-4}$ S/cm)

This candidate exhibits highly competitive predicted room-temperature conductivity while remaining within the model's validated chemical domain.
