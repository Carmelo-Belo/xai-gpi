
# 🌪️ Spatial Cluster-Optimized Feature Selection (SCO-FS) for Tropical Cyclone Genesis Detection

This repository contains the code, data processing pipeline, and results for the paper:

**"Improving Tropical Cyclone Genesis Detection with Feature Selection and Explainable Artificial Intelligence"**

---

## 📘 Overview

Tropical cyclone genesis (TCG) remains one of the key challenges in climate and weather science. This project introduces a novel machine learning framework—**Spatial Cluster-Optimized Feature Selection (SCO-FS)**—to improve long-term detection and interpretation of cyclone formation using environmental variables and climate indices.

SCO-FS is applied across six major tropical ocean basins and incorporates:
- Spatial clustering to reduce dimensionality,
- Evolutionary optimization for feature selection,
- Neural networks for prediction,
- SHAP (SHapley Additive exPlanations) for model interpretability.

---

## 🧠 Key Features

- 🌍 Supports **six ocean basins**: North Atlantic, Northeastern Pacific, Northwestern Pacific, North Indian, South Indian, and South Pacific.
- 📊 Incorporates reanalysis data (e.g., ERA5) and climate indices (e.g., Niño3.4, SOI, PDO).
- ⚙️ Uses **Probabilistic Coral Reef Optimization with Substrate Layers (PCRO-SL)** for robust feature selection.
- 🧠 Neural network models are trained with features selected per basin.
- 🔍 SHAP-based explainability to explore **impacts** of environmental drivers.

---

## 🌿 Branches

- `main`: clean version of the code for consultation (publication version)
- `dev`: full development history and work in progress

Feel free to fork the repo and explore both branches.
Pull requests are welcome via the `dev` branch.

---

## 📁 Repository Structure

```
.
├── best_model_analysis         # Notebooks and code for training final models and perform final analysis
|   ├── figures/                # Figures showing results from final analysis
├── data/                       # Datasets for the final model, GPIs time series, and code for dataset construction. 
├── clustering/                 # Clustering code
├── PyCROSL/                    # Code of the Probabilistic Coral Reef Optimization with Substrate Layers algorithm
├── results/                    # Final models files and information, tabular data containing info on all simulations performances and final selected features for the best models
├── CRO_Spatio_FS_PI.py/        # Code to perform feature selection with Physicall-Informed Light Gradiend Boosting Machine
├── CRO_Spatio_FS.py/           # Code to perform feature selection with Linear Regression or Light Gradiend Boosting Machine
├── test_results_analysis/      # Code to run the *evalNN* training for a specific simulation
├── requirements.txt            # Python dependencies
├── LICENSE                     # License file
└── README.md                   # This file
```

---

## 🚀 Getting Started

1. **Clone the repository:**

```bash
https://github.com/Carmelo-Belo/tcg_fsM.git
cd tcg_fsM
```

2. **Set up the virtual environment:**

```bash
# Create the conda environment
conda env create -f environment.yml

# Acticate the conda environment
conda activate tf_tcg
```

---

## 📊 Results Summary

- SCO-FS models outperform state-of-the-art Genesis Potential Indices (GPIs) in capturing interannual TCG variability.
- SHAP analysis reveals physically meaningful drivers, especially for **North Atlantic** and **Northeastern Pacific**.
- Insights gained into year-to-year cyclone activity changes.

---

## 📄 Related Paper

> *Improving Tropical Cyclone Genesis Detection with Feature Selection and Explainable Artificial Intelligence*  
> Authors: [Your Name(s)]  
> [Journal Name, Year]  
> [DOI or Preprint Link]

---

## 🤝 Contributing

If you find this work useful or would like to contribute improvements, feel free to fork the repo or open an issue. Suggestions are welcome!

For collaboration or any queries regarding this project, please contact:
- Filippo Dainelli: [filippo.dainelli@polimi.it]

---

## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👀 Additional information

Please refer to the following links for more information regarding the Probabilistic Coral Reef Optimization with Substrate Layers (PyCROSL) algorithm and the original application of the SpatioTemporal Cluster-Optimized Feature Selection (STCO-FS):

- [PyCROSL](https://github.com/jperezaracil/PyCROSL)
- [STCO-FS](https://github.com/GheodeAI/STCO-FS)