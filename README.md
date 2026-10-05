# Machine Learning-Assisted Discovery of Next-Generation Solid-State Battery Materials

## Scientific Objective
The primary objective of this project is to evaluate the applicability boundaries of composition-only machine learning models in predicting the room-temperature ionic conductivity of solid-state electrolytes. By strictly defining the physical limitations of our predictive models, we aim to establish a rigorous, mathematically sound pipeline for computational materials screening.

## Methodological Rigor
This pipeline ensures absolute scientific integrity and prevents data leakage:
- **Dataset Curation:** Built upon the experimental OBELiX solid-state electrolyte dataset.
- **Duplicate Aggregation:** Formulas exhibiting high physical variance (>100x variance in conductivity measurements across labs/polymorphs) were explicitly dropped, while low-variance duplicates were safely median-aggregated in log-space.
- **Strict Cross-Validation:** A custom scikit-learn `Pipeline` (integrating imputation and standardization) was placed inside the CV loop to dynamically recalculate statistics exclusively on training folds, entirely eliminating data leakage.

## Key Findings
- **Chemical Domain Overfitting:** Standard random cross-validation produced a deceptively strong $R^2 \approx 0.67$. However, implementing a stringent `GroupKFold` strategy—where the model was tested entirely on held-out anion families (true zero-shot extrapolation)—revealed a collapse in performance ($R^2 \approx 0.11$). This mathematically proved that composition-only features cannot safely extrapolate conductivity predictions across radically different crystal chemistry families.
- **Domain-Restricted Screening:** Acknowledging the above limitation, we constrained our subsequent Materials Project (MP) theoretical search exclusively to validated domains (Oxides and Sulfides). The model successfully identified **$LiLa_{5}Ti_{8}O_{24}$** (a Lithium Lanthanum Titanate, LLTO) as a highly promising candidate, featuring near thermodynamic stability (energy above hull = 0.047 eV/atom) and a highly competitive predicted room-temperature conductivity of $\approx 1.65 \times 10^{-4}$ S/cm.

## Repository Structure
```text
.
├── data/
│   ├── interim/           # Cleaned dataset (Not tracked by Git)
│   ├── processed/         # Featurized, ranked, and explainability artifacts
│   └── raw/               # Downloaded CSVs and API candidate batches
├── src/
│   ├── models/            # Core ML pipelines, CV strategies, and explainability scripts
│   ├── mp_screening/      # Materials Project API integration and prediction scripts
│   └── ...                # EDA, downloading, and feature engineering modules
├── tests/                 # Comprehensive Pytest suites guaranteeing pipeline stability
├── research_conclusions.md# Formal summary of research outcomes
└── research_log.md        # Detailed developmental log of the research phases
```

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set up your environment variables:
   Create a `.env` file in the root directory and add your Materials Project API key:
   ```text
   MP_API_KEY=your_key_here
   ```
3. Run the automated test suite to ensure pipeline integrity:
   ```bash
   pytest tests/
   ```
