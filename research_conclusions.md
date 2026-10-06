# Solid-State Battery Materials: Machine Learning Screening Conclusions

## Project Objective
The primary objective of this project was to predict the room-temperature ionic conductivity of lithium-ion solid electrolytes using strictly composition-based Magpie features, enabling rapid computational screening of potentially synthesizable candidates from the Materials Project based on heuristic stability criteria.

## Methodological Rigor
To ensure physical reliability and eliminate data leakage, we implemented a rigorous data processing and validation pipeline:
- **Duplicate Aggregation:** We identified identical chemical formulas in the raw dataset with over 1.0 order of magnitude variance in measured conductivity (due to polymorphism or inter-lab variance). 18 unstable labels were explicitly dropped, and low-variance duplicates were aggregated via their median log-conductivity. This necessary aggregation inherently results in the loss of specific experimental nuances (such as synthesis method, temperature, or grain boundary effects) mapped to identical formulas, ultimately forming a baseline of exactly 500 unique materials.
- **Applicability Domain Constraints:** Using a stringent `GroupKFold` validation strategy based on the primary anion, we uncovered severe **Chemical Domain Overfitting**. While random cross-validation achieved an $R^2$ of ~0.67, zero-shot extrapolation to unseen anion families dropped the $R^2$ to ~0.11. Consequently, our final candidate screening was strictly confined to known applicability domains using a 95th-percentile Nearest Neighbors distance filter to prevent unsafe theoretical extrapolation.
- **Uncertainty Quantification (UQ):** Final candidates were ranked using a heuristic Lower Bound (LCB), dynamically subtracting the standard deviation of the Random Forest tree ensemble from the mean prediction. This strictly penalizes volatile predictions.

## Feature Drivers
By opening the Random Forest "black box" (via native Gini feature importances), we identified the primary compositional drivers of conductivity:
1. **`MagpieData mean CovalentRadius`:** The dominant predictor. Because this is a strict composition-only model, this feature is purely a statistical aggregation of elemental covalent radii. It acts as a compositional marker separating chemical families, but it does NOT contain or learn actual crystallographic structural data, grain boundaries, or lithium-ion migration bottlenecks.
2. **`MagpieData avg_dev SpaceGroupNumber`:** This is a composition-only model. Features such as `SpaceGroupNumber` represent composition-weighted averages of the elemental standard states, NOT the true crystallographic structure or symmetry of the compound. While mathematically predictive, this repository explicitly acknowledges that true structural information (such as distinct polymorphs or specific lattice arrangements) cannot be resolved by these elemental proxy features.
3. **`MagpieData mean Electronegativity`:** A known proxy for lattice polarizability. A softer, more polarizable anion framework (lower average electronegativity) typically weakens the electrostatic binding between the lithium ion and the lattice, facilitating higher mobility.

*Caveat:* While these features possess strong physical intuition, they represent predictive correlations rather than continuous physical laws. In composition-only models, features like electronegativity may partially act as mathematical proxies separating distinct chemical families (e.g., oxides vs. sulfides) rather than capturing continuous physical mechanisms.

## Top Candidate Identification
After training on the complete, cleaned dataset, we executed a domain-restricted query against the modern Materials Project API to retrieve computationally stable and metastable, lithium-containing candidates. 

Our pipeline flagged **$LiLa_{5}Ti_{8}O_{24}$** as a primary candidate for further investigation. Notably, this composition belongs to the well-known Lithium Lanthanum Titanate (LLTO) family of perovskite-type solid electrolytes. The fact that the model independently selected a known high-conductivity chemical space from a raw database query provides supporting physical context and plausibility for the model's relevance, rather than claiming the invention of a completely new chemistry or constituting independent experimental validation.
- **Energy Above Hull:** 0.047 eV/atom (metastable; not thermodynamically stable at 0K, though potentially synthesizable in practice)
- **Predicted log10(Conductivity):** -4.03 (~$9.33 \times 10^{-5}$ S/cm)

This candidate exhibits highly competitive predicted room-temperature conductivity while remaining safely within the model's validated chemical domain.
