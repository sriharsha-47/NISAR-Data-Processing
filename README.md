# Unsupervised Domain Adaptation for Cross-Domain Classification of Mangrove Species

**Internship Research Project | Remote Sensing | Machine Learning | Unsupervised Domain Adaptation**

## Overview

This project investigates unsupervised domain adaptation techniques for cross-domain mangrove species classification using preprocessed remote-sensing feature embeddings.

Machine-learning models trained on one geographic region may experience reduced performance when applied to another region because of differences in feature distributions, environmental conditions, and geographic characteristics. This project explores feature alignment techniques to address this domain shift and evaluate their usefulness for transferring mangrove classification models between study areas.

The experimental workflow combines feature-space alignment, Random Forest classification, spatial prediction, probability-map generation, and quantitative evaluation.

The implementation and experiments were developed using Python and Jupyter notebooks.

## Research Objectives

- Investigate domain shift between geographically distinct mangrove regions.
- Study CORAL and Subspace Alignment for aligning source and target feature distributions.
- Evaluate hybrid approaches combining multiple alignment techniques.
- Apply Random Forest classifiers to mangrove species classification.
- Generate spatial classification and class-probability maps.
- Compare cross-domain performance with conventional classification baselines.
- Examine feature-distribution differences and classification performance using quantitative metrics and visualizations.

## Study Areas

The project considers three mangrove study areas:

| Study area | State |
|---|---|
| Krishna Delta | Andhra Pradesh |
| Malad Creek | Maharashtra |
| Pichavaram | Tamil Nadu |

The experiments investigate classification transfer across geographic domains and include a Pichavaram Random Forest baseline.

## Methodology

### 1. Feature Preparation and Domain Analysis

The project uses preprocessed 64-band feature embeddings supplied for the internship experiments. Feature distributions are examined to understand differences between source and target domains.

Temporal alignment experiments also investigate feature differences between 2019 and 2024 data, including stable-pixel selection and covariance analysis.

### 2. CORAL — Correlation Alignment

CORAL reduces differences between source and target feature covariance structures through a linear feature transformation.

The project includes CORAL-related implementations and experiments for feature alignment and cross-domain classification.

### 3. Subspace Alignment (SA)

Subspace Alignment uses principal component analysis to obtain source and target subspaces and aligns the source subspace with the target subspace.

Experiments investigate subspace distance, different subspace dimensions, and the resulting feature representations.

### 4. Hybrid Alignment Methods

The project explores combinations of CORAL and Subspace Alignment:

- CORAL followed by Subspace Alignment.
- Subspace Alignment followed by CORAL.

These experiments investigate whether combining covariance alignment and subspace alignment improves cross-domain feature compatibility.

### 5. Transfer Component Analysis

The repository also contains exploratory work involving Transfer Component Analysis (TCA), a feature-based domain adaptation approach.

### 6. Random Forest Classification

Random Forest models are used for mangrove species classification. The classification experiments include cross-domain prediction and a Pichavaram baseline.

### 7. Spatial Prediction and Evaluation

The workflow includes spatial predictions, classification maps, probability maps, confusion-matrix analysis, and evaluation against available reference labels.

The reported evaluation metrics include Overall Accuracy and Cohen's Kappa.

## Experimental Evaluation

The project examines:

- Cross-domain classification performance.
- Comparison between source and target feature distributions.
- Covariance differences before and after alignment.
- Subspace alignment behavior across different dimensions.
- Spatial classification and probability-map outputs.
- Baseline performance within a geographic domain.
- Agreement between predictions and reference labels.

Quantitative results should be interpreted in the context of the specific domain transfer, reference labels, and evaluation procedure used in each experiment.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Numerical computing | NumPy, SciPy |
| Data manipulation | Pandas |
| Machine learning | Scikit-learn |
| Geospatial processing | Rasterio, GeoPandas |
| Visualization | Matplotlib, Seaborn |
| Dimensionality reduction | Scikit-learn PCA |
| Domain adaptation | CORAL, Subspace Alignment, exploratory TCA |
| Experimental environment | Jupyter Notebook |
| Model and artifact handling | Pickle, Joblib |

The precise dependencies vary across notebooks. Refer to `requirements.txt` and the imports in individual notebooks for the libraries used by each experiment.

## Full Project Repository Structure

The following structure represents the main directories and files in the original project workspace. Individual generated artifacts and geospatial data files are organized within these directories.

```text
NISAR_Data_project/
│
├── .ipynb_checkpoints/
│
├── classifier models/
│   └── Models/
│       └── Trained classifiers and associated prediction artifacts
│
├── Data/
│   ├── Krishna-region raster datasets
│   ├── Malad-region reference data
│   ├── Pichavaram-region raster and reference data
│   ├── 2019 and 2024 feature datasets
│   └── Shapefiles and other geospatial inputs
│
├── Notefiles/
│   ├── Learning_notebook.ipynb
│   ├── previous work.ipynb
│   ├── binary_classifier_learning.ipynb
│   ├── cross_domian_learning.ipynb
│   ├── cross_domain_adaptation.ipynb
│   ├── binary_classifier_malad.ipynb
│   ├── Binary_classifier_pichavaram.ipynb
│   ├── checking_predtiction_pichavaram.ipynb
│   ├── Subspace_alignment.ipynb
│   ├── Transfer_component_analysis.ipynb
│   ├── Classifier_Model.ipynb
│   └── Untitled.ipynb
│
├── Outputs/
│   ├── CORAL alignment artifacts
│   ├── Subspace Alignment artifacts
│   ├── Intermediate geospatial outputs
│   └── Alignment metadata and experiment outputs
│
├── output_plots/
│   ├── Experimental visualizations
│   ├── Alignment comparison plots
│   ├── Dimension-sweep results
│   ├── Metrics and tabular outputs
│   └── Classification artifacts
│
├── Pichavaram_RF_Baseline/
│   ├── Accuracy_Assessment/
│   ├── Figures/
│   ├── Final_Classification/
│   ├── Models/
│   └── Probability_maps/
│
├── aligned_2019_to_2024.tif
├── internship_report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** This is a structured overview of the known project contents, not a guaranteed file-by-file inventory. The folders contain additional artifacts, and their exact contents may change as experiments evolve.

## Notebook Directory Guide

The `Notefiles/` directory contains exploratory experiments, alignment implementations, classification workflows, and evaluation notebooks.

| Notebook | Main focus |
|---|---|
| `Learning_notebook.ipynb` | Raster inspection, temporal feature comparison, stable-pixel analysis, and CORAL experiments |
| `previous work.ipynb` | Temporal domain alignment, stable-pixel selection, covariance analysis, and visualization |
| `binary_classifier_learning.ipynb` | Binary classification experiments |
| `cross_domian_learning.ipynb` | Feature analysis and CORAL-based cross-domain experiments |
| `cross_domain_adaptation.ipynb` | Cross-domain adaptation and classification experiments |
| `binary_classifier_malad.ipynb` | Malad-related binary classification experiments |
| `Binary_classifier_pichavaram.ipynb` | Pichavaram-related binary classification experiments |
| `checking_predtiction_pichavaram.ipynb` | Pichavaram prediction inspection |
| `Subspace_alignment.ipynb` | PCA-based Subspace Alignment, CORAL comparisons, hybrid methods, and TCA experiments |
| `Transfer_component_analysis.ipynb` | Exploratory Transfer Component Analysis work |
| `Classifier_Model.ipynb` | Classifier-related experiments |
| `Untitled.ipynb` | Additional exploratory notebook |

Notebook descriptions summarize their apparent focus; they should not be interpreted as a guarantee that every notebook represents a finalized experiment.

## Project Artifacts

### `Data/`

Contains the project's geospatial inputs, including large raster datasets, reference data, and shapefiles used by the experimental workflows.

### `classifier models/`

Contains saved classification models and related prediction artifacts.

### `Outputs/`

Stores intermediate outputs from feature alignment and related experiments, including alignment metadata and geospatial artifacts.

### `output_plots/`

Contains experimental visualizations, alignment comparisons, dimension-sweep results, and selected tabular or classification outputs.

### `Pichavaram_RF_Baseline/`

Organizes the Pichavaram Random Forest baseline artifacts into accuracy assessment, figures, final classification, models, and probability maps.

### `aligned_2019_to_2024.tif`

A raster artifact associated with temporal alignment experiments.

### `requirements.txt`

Lists the Python dependencies declared for the project. Individual notebooks may require additional libraries or environment-specific configuration.

### `internship_report.pdf`

Contains internship documentation. This file is part of the original workspace and should only be included in a public repository if its publication is explicitly authorized.

## Results and Interpretation

The experiments investigate whether feature alignment can reduce source–target distribution differences and support cross-domain classification.

The project considers both quantitative evaluation and spatial outputs. However, alignment alone does not guarantee improved classification performance: the outcome depends on the selected source and target domains, class distributions, feature representations, and evaluation methodology.

Reported metrics should therefore be associated with their corresponding experiments rather than interpreted as a universal performance guarantee.

## Scope and Limitations

- The project uses preprocessed feature embeddings rather than implementing the complete raw satellite-data processing pipeline.
- Performance can vary substantially across geographic transfers.
- Reference-label availability and class imbalance can affect evaluation.
- Exploratory notebooks may contain intermediate experiments and outputs.
- Some notebooks depend on locally stored data and environment-specific paths.
- The public code showcase is not intended to be a fully reproducible research package.

## Confidentiality and Public Release

This project was developed as part of an internship. The public repository should contain only material authorized for disclosure.

Large geospatial datasets, trained models, intermediate outputs, and internship documentation may be subject to access restrictions. Their presence in the original local workspace does not imply permission to publish them.

The `.gitignore` file is intended to reduce accidental uploads of large and generated artifacts. It is not a substitute for reviewing staged files before committing.

## Acknowledgment

Developed as part of an internship research project on unsupervised domain adaptation for cross-domain mangrove species classification.

Institutional attribution, acknowledgments, and publication details should follow the applicable internship and confidentiality guidelines.
