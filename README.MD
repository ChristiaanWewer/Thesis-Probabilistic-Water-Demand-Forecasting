
# Codebase for Master's Thesis: *Exploring Probabilistic Short-Term Water Demand Forecasts at the District Level Using Neural Networks*

This repository accompanies my master's thesis, focusing on enhancing reproducibility and enabling the community to quickly apply the methods used in my research. The methods can be re-applied to different datasets, and experiments can easily be adapted, for instance, by adding new features.

For any questions or feedback, feel free to contact me at **[christiaanwewer@gmail.com](mailto:christiaanwewer@gmail.com)**.

---

## 🚀 Installation

To set up and use this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/water-demand-forecasting.git
   cd water-demand-forecasting
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Codebase Overview

The codebase is structured as follows, with all code well-commented for easier understanding:

### `data/` 📊

This folder contains subfolders for managing data at various stages of the pipeline:
- **`raw/`**: Stores the raw data from the [Battle of Water Demand](https://wdsa-ccwi2024.it/battle-of-water-networks/).
- **`processed/`**: Contains:
  - Results from the naive model.
  - Splits for training, validation, and testing.
  - Scalers used for normalization.
- **`sequences/`**: Stores the sequences generated for training the neural network models.

### `data_investigation/` 🔍

Contains notebooks and utility files for data analysis and preparation:
- **`benchmark_model.ipynb`**: Implementation of the benchmark model used in this research.
- **`create_sequences.ipynb`**: 
  - Generates sequences for model training from any Pandas DataFrame.
  - Converts data into PyTorch tensors using dictionaries for training, a key is a (potential) feature in the training data.
  - Can turn any Pandas Column into a feature
  - Optionally creates one-hot encoded features.
- **`data_investigation.ipynb`**: Includes the data analysis conducted for the thesis.
- **`utils.py`**: Helper functions for sequence creation.

### `deterministic_models/` 🤖

Contains Python scripts for training deterministic models.

### `probabilistic_models/` 📈

Contains Python scripts for training probabilistic models using **Weights & Biases (wandb)**.

### `show_results/` 📊

Includes two notebooks:
- **`results_point_forecasts.ipynb`**:
  - Downloads models from wandb.
  - Runs final checks.
  - Plots results for point forecasts.
- **`results_probabilistic_forecasts.ipynb`**:
  - Similar workflow for probabilistic forecast models.

### `ForecastingModel/` 🛠️

The core codebase for model training and evaluation:
- **`Forecaster.py`**:
  - Implements the `Forecaster` class for training neural networks.
  - Example usage: `deterministic_models/MLP_Vanilla.py`.
  - Supports dropout fitting (via dichotomic search) and Monte Carlo Dropout.
  - Example usage: `probabilistic_models/MCD_MLP.py`.
- **`ConformalPrediction.py`**:
  - Wraps the `Forecaster` class to compute Conformalized Quantile Regression (CQR) models.
  - Example usage: `probabilistic_models/MLP_CQR.py`.
- **`DownloadRuns.py`**:
  - Utility functions for downloading projects from wandb.
- **`ScoreFunctions.py`**:
  - Includes all scoring metrics used in the project.
- **`ProbabilisticLosses.py`**:
  - Implements loss functions like:
    - Quantile Loss (Pinball Loss).
    - Negative Likelihood Loss (used for Mixture Density Networks).

---

### `figures/` 📷

This folder contains visualizations and plots used in the thesis. It includes:
- Final plots generated during the analysis.

