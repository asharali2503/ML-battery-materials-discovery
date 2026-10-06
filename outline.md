# Title: Evaluating the Limits of Composition-Only Machine Learning for Solid-State Electrolytes: A Case Study in Applicability Domains and Uncertainty Quantification

## Abstract
- Briefly introduce the challenge of rapid computational screening for solid-state electrolytes using composition-only descriptors.
- State the rigorous dataset baseline: 500 robustly cleaned experimental observations, dropping 18 highly ambiguous labels.
- Highlight the core methodological finding: Standard cross-validation overestimates performance ($R^2 \approx 0.67$) due to chemical similarity, while rigorous out-of-domain evaluation (`GroupKFold`) reveals a collapse in generalization ($R^2 \approx 0.11$).
- Conclude with the successful application of Applicability Domain (AD) filtering and Lower Confidence Bound (LCB) ranking to recover the well-known Lithium Lanthanum Titanate (LLTO) family ($LiLa_5Ti_8O_{24}$), validating the pipeline's physical intuition without claiming novel discovery.

## Introduction
- Discuss the trade-off in materials informatics between high-throughput composition-only models and computationally expensive structure-based models (e.g., CGCNNs).
- Review literature on the limitations of composition models, specifically their inability to resolve distinct crystallographic polymorphs, bottleneck sizes, and 3D percolation networks.
- Introduce the concept of "Chemical Domain Overfitting" and the necessity of spatial or cluster-based cross-validation to prevent data leakage.
- Outline the paper's objective: To build a mathematically leak-proof ML pipeline that strictly enforces Applicability Domains (AD) and Uncertainty Quantification (UQ) to prevent extrapolative hallucinations in candidate screening.

## Methodology
- **Dataset Curation & Aggregation:** Detail the preprocessing of the OBELiX dataset, focusing on the 1.0 log-conductivity variance threshold used to filter unstable labels and the log-space median aggregation of duplicates.
- **Leak-Proof Cross-Validation:** Explain the implementation of `sklearn.pipeline.Pipeline` within the custom CV loop to isolate `SimpleImputer` and `StandardScaler` transformations, completely preventing target leakage.
- **Group-Based Validation:** Describe the `GroupKFold` strategy, defining the 'groups' by the primary (most electronegative) anion to test true zero-shot chemical extrapolation.
- **AD and UQ Filtering:** Detail the 95th-percentile `NearestNeighbors` distance metric used to define the training boundary, and the Lower Confidence Bound (LCB) calculation (mean prediction minus Random Forest tree standard deviation) used to penalize uncertain predictions.

## Results
- **Validation Metrics:** Report the stark contrast between Random 5-Fold CV (MAE: 1.23, $R^2$: 0.67) and GroupKFold CV (MAE: 1.99, $R^2$: 0.11).
- **Feature Importance:** Present the top predictive features (`MagpieData mean CovalentRadius`), interpreting them as mathematical proxies for unresolvable structural metrics like interstitial volume.
- **Candidate Screening:** Describe the query against the Materials Project API (yielding 50 stable/metastable candidates within known domains) and report that 18 were rigorously filtered out by the AD distance threshold (6.98).
- **Candidate Ranking:** Present the top surviving candidate, $LiLa_5Ti_8O_{24}$, with its mean prediction ($\approx 9.33 \times 10^{-5}$ S/cm), standard deviation (2.01), conservative LCB score (-6.04), and exact Energy Above Hull (0.047 eV/atom).

## Discussion
- **Physical Validation via LLTO:** Contextualize the model's top pick, noting that $LiLa_5Ti_8O_{24}$ is stoichiometrically equivalent to $Li_{0.125}La_{0.625}TiO_3$, a heavily studied LLTO perovskite.
- **Accuracy of the Prediction:** Compare the model's $\sim 10^{-5}$ S/cm prediction to experimental literature, confirming that while LLTO bulk conductivity is higher ($\sim 10^{-3}$ S/cm), the model accurately predicted the practical, grain-boundary-limited *total* conductivity.
- **Interpreting Uncertainty:** Discuss the physical meaning of the large prediction standard deviation (2.01). Argue that this variance accurately reflects the model's structural ignorance; without XRD data, the ensemble outputs a wide confidence interval to account for unknown polymorphic phases.
- **Limitations:** Reiterate that composition-only models are best used as rough, first-pass filters. Emphasize that the 0.047 eV/atom metastability aligns with known synthesis challenges (e.g., lithium volatilization and Ti reduction).

## Conclusion
- Summarize that while composition-only models cannot safely extrapolate into unknown chemistry, they can successfully identify high-performing candidates when strictly bound by AD and UQ constraints.
- Emphasize that recovering the LLTO family from a raw database query acts as a profound sanity check for the ML pipeline.
- Propose that future work must integrate structural descriptors or textual LLM metadata to bridge the gap between compositional speed and structural accuracy.
