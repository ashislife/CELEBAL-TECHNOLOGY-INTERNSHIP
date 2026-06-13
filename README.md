# Celebal Technology Internship - Data Science 

Welcome to my repository for the **Celebal Technology Internship**. This repository serves as a tracker for my weekly assignments, projects, and learnings during the internship program.

## 📌 Overview: Week 1 Assignment
The first week focuses on building a solid foundation in mathematics, programming, and data manipulation libraries required for Machine Learning.

* **File:** `week1_Ashish_Kumar.ipynb`
* **Environment:** Created and executed using Google Colab / Jupyter Notebooks.
* **Status:** 🏁 All tasks completed, assertion blocks passed successfully, and visualizations rendered.

---

## 🛠️ Topics Covered & Implementations

### Part 1: Python Fundamentals
* **Control Flow:** Implemented conditional logic to classify numbers based on specific ranges.
* **Data Structures:** Work with loops, sets, and list comprehensions to filter data and count frequencies without using external libraries like `Counter`.
* **Exception Handling:** Built robust functions with custom `try-except` blocks to handle errors like `ZeroDivisionError` and `TypeError`.
* **Lambdas & Higher-Order Functions:** Demonstrated functional programming concepts using custom lambda functions.

### Part 2: NumPy (Numerical Python)
* **Array Manipulation:** Array creation, multi-dimensional reshaping (1D to 2D and 3D), and checking metadata (`shape`, `ndim`, `dtype`).
* **Slicing & Indexing:** Advanced indexing and boolean masking to filter array elements.
* **Vector & Matrix Math:** Implementation of element-wise multiplication, scalar operations, and matrix dot products (`@` operator).

### Part 3: Pandas (Data Manipulation)
* **Data Structures:** Difference and usage of Pandas `Series` and `DataFrames`.
* **Data Selection:** Practical usage of `.loc` and `.iloc` indexing.
* **Aggregation:** Filtering datasets and using `.groupby()` along with `.agg()` to find department-wise stats.
* **Data Cleaning:** Handling missing (NaN) values using median/mean imputation and dropping incomplete rows.

### Part 4: Linear Algebra
* **Vector Visualization:** Computing $L_2$ norm (Euclidean distance) and plotting vectors using `matplotlib`.
* **Matrix Properties:** Demonstrating matrix addition, scalar multiplication, and proving that matrix multiplication is non-commutative ($P \cdot Q \neq Q \cdot P$).
* **Eigenvalues & Eigenvectors:** Computing eigenpairs using `np.linalg.eig`, validating the characteristic equation $A \mathbf{v} = \lambda \mathbf{v}$, and understanding its geometric stretching/compressing meaning.
* **SVD (Singular Value Decomposition):** Decomposing matrices, performing rank-1 approximations, and understanding its direct connection to PCA (Principal Component Analysis).

### Part 5: Statistics
* **Descriptive Statistics:** Calculating mean, median, standard deviation, and IQR, paired with KDE (Kernel Density Estimate) histogram plots.
* **Hypothesis Testing:** Conducting a One-Sample T-Test to check statistical significance against a population mean and interpreting Pearson correlation ($r$).
* **Error Metrics from Scratch:** Implementing MAE, MSE, RMSE, $R^2$, and Adjusted $R^2$ scores without using `scikit-learn`.
* **Stationarity & Distribution Testing:** Utilizing Kolmogorov-Smirnov (KS) test for normality and Augmented Dickey-Fuller (ADF) test for time-series stationarity (with differencing).
* **Model Monitoring:** Coding the Population Stability Index (PSI) from scratch to detect and visualize Covariate/Concept drift.

### Part 6: Probability Theory
* **Core Concepts:** Calculating joint and conditional probabilities, and mathematically proving independent vs. dependent events.
* **Distributions:** Plotting and analyzing Normal, Binomial, and Poisson distributions alongside their ML use cases.
* **Bayes' Theorem:** Building a Naive Bayes prediction workflow for spam filtering and mapping terms (Prior, Likelihood, Evidence, Posterior).
* **Central Limit Theorem (CLT):** Simulating sample means from an Exponential distribution to visually and statistically (KS-test) prove the convergence towards a normal distribution.

---

## 🚀 How to Run the Notebook
1. Clone the repository:
   ```bash
   git clone [https://github.com/ashislife/CELEBAL-TECHNOLOGY-INTERNSHIP.git](https://github.com/ashislife/CELEBAL-TECHNOLOGY-INTERNSHIP.git)

   Install the required dependencies:

Bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn joblib
Open Jupyter Notebook/Lab or upload the files directly to Google Colab / Kaggle.

Execute the cells sequentially to reproduce the analysis, visualizations, and model results.




------------------------------------------------<>-------------------------------------


   # 📌 Week 2 Assignment: End-to-End ML Regression Pipeline

This folder contains the complete Machine Learning pipeline developed during **Week 2** of the Celebal Technology Internship. The project utilizes historical Tesla data to build a predictive regression model.

* **Notebook File:** `week2-ashish-kumar-ipynb.ipynb`
* **Dataset Used:** Tesla Deliveries and Production Data (2015 - 2025)
* **Environment:** Kaggle / Jupyter Notebook
* **Final Model Accuracy:** **$R^2$ Score ≈ 0.989** 🏁

---

## 🛠️ Pipeline Steps & Implementation

### 1. Data Loading & Understanding
* Imported core analytics libraries like `numpy` and `pandas`.
* Loaded the Tesla dataset containing 2,640 records and 12 columns (including `Year`, `Model`, `Production_Units`, `Avg_Price_USD`, `CO2_Saved_tons`, etc.).
* Inspected dataset structure using `df.info()` and evaluated basic statistics via `df.describe()`.

### 2. Exploratory Data Analysis (EDA) & Visualization
* **Production vs Deliveries:** Created a scatter plot using `seaborn` to visualize the relationship and density between production units and estimated deliveries.
* **Correlation Heatmap:** Generated a full correlation matrix heatmap to check multi-collinearity and dependencies among numerical features.

### 3. Data Preprocessing & Feature Engineering
* **Categorical Encoding:** Converted categorical attributes (`Region`, `Model`, `Source_Type`) into numerical integers using `LabelEncoder` from `scikit-learn`.
* **Feature Selection:** Split the dataset into features ($X$) and target variable ($y$ - `Estimated_Deliveries`).
* **Train-Test Split:** Segmented the data into **80% Training data** and **20% Testing data** using `train_test_split` with a fixed `random_state=42` for reproducibility.

### 4. Model Building & Training
* Implemented a **Random Forest Regressor** (`n_estimators=100`) to capture complex non-linear trends and minimize the risk of overfitting.

### 5. Model Evaluation & Performance
The model's predictions on unseen test data were evaluated using standard regression metrics:
* **MAE (Mean Absolute Error):** ~323.46
* **RMSE (Root Mean Squared Error):** ~409.50
* **$R^2$ Score:** **0.9887** (Indicates that the model successfully explains 98.8% of the variance in Tesla's estimated deliveries).

### 6. Feature Importance Analysis
* Extracted feature importances from the trained Random Forest model and plotted them using a bar chart.
* **Key Insight:** `Production_Units` emerged as the most dominant feature by a huge margin, contributing to ~99% of the model's decision-making process.

### 7. Model Deployment Preparation
* Successfully serialized and saved the final trained model object as `tesla_delivery_model.pkl` using the `joblib` library for instant future inference.


```bash
## 🚀 Dependencies & How to Run
Make sure you have the following packages installed before running the notebook:

pip install numpy pandas matplotlib seaborn scikit-learn joblib

Open the file week2-ashish-kumar-ipynb.ipynb in Jupyter Notebook, Lab, or Kaggle.

Run all cells sequentially to recreate the data analysis, visualizations, and model training.
```


-------------------------------------------<>------------------------------------------


# 📌 Week 3 Assignment: Develop a Customer Intelligence System using Classification, Ensemble Learning & Clustering

This folder contains the complete end-to-end Machine Learning workflow developed during **Week 3** of the Celebal Technology Internship. The project focuses on building an intelligent customer segmentation and classification system based on socio-economic metrics.

* **Notebook File:** `week3-ashish-kumar-ipyn.ipynb`
* **Dataset Used:** Unsupervised Learning Dataset (167 Records, 10 Features)
* **Environment:** Kaggle / Jupyter Notebook
* **Best Performing Classifier:** **Random Forest Classifier (100% Accuracy)** 🏁

---

## 🛠️ Pipeline Steps & Implementation

### 1. Data Loading & Understanding
* Loaded and inspected the dataset using `df.head()`, `df.info()`, and `df.describe()` to map out the distribution of numerical attributes.
* Checked dataset dimensions which contains 167 rows and 10 structural features.

### 2. Exploratory Data Analysis (EDA) & Correlation
* **Correlation Heatmap:** Plotted a comprehensive correlation matrix using `seaborn` to check dependencies.
* **Key Insights:** * Strong positive correlation discovered between `income` and `gdpp`.
  * `child_mort` shows an extreme negative correlation with `life_expec` and `income`, proving to be a high-variance indicator for segmentation.

### 3. Data Preprocessing & Feature Scaling
* Dropped the unique categorical `country` column since it works as an identifier and doesn't add value to mathematical distance metrics.
* Normalized all remaining numerical features using `StandardScaler` to prevent high-magnitude features (like GDP/Income) from biasedly dominating the clustering algorithms.

### 4. K-Means Clustering (Segmentation)
* **Elbow Method:** Ran iterations from $K=1$ to $10$ plotting the Within-Cluster Sum of Squares (WCSS) to locate the ideal elbow point.
* **Model Fitting:** Selected **$K = 3$** as the optimal configuration to segment the data into three distinct operational profiles:
  * **Cluster 0 (Developed):** High income, high GDP, and minimal child mortality rate.
  * **Cluster 1 (Underdeveloped):** Low income, low GDP, and severe child mortality rate.
  * **Cluster 2 (Developing):** Average economic indicators and stable health metrics.

### 5. Density-Based Clustering (DBSCAN)
* Implemented DBSCAN to understand the density patterns and separate structural core groups from noisy outliers.
* Conducted hyperparameter tuning over `eps` and `min_samples` ranges, finalizing **`eps=1.2`** and **`min_samples=3`** to extract meaningful density clusters while isolating extreme anomalies.

### 6. Classification & Ensemble Learning
Using the optimal K-Means cluster categories as the ground-truth target variable ($y$), the data was split into an **80% Training / 20% Testing** partition using stratified sampling:
* **Random Forest Classifier:** Trained an ensemble of 100 decision trees. Achieved an absolute perfect **100% Accuracy** on unseen test data.
* **XGBoost Classifier:** Implemented gradient boosting to benchmark performance, yielding a strong **94.12% Accuracy**.

### 7. Feature Importance Analysis
* Derived the underlying feature importance matrix from the top-performing Random Forest model.
* **Top Drivers:** `child_mort` (25.8%) and `gdpp` (19.7%) emerged as the primary split drivers determining the customer intelligence classification, followed closely by `total_fer` and `life_expec`.

---

## 📊 Model Performance Comparison

| Model | Test Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **100.00%** | **1.00** | **1.00** | **1.00** |
| **XGBoost Classifier** | **94.12%** | **0.95** | **0.94** | **0.94** |

---

## 🚀 Dependencies & Installation
Ensure you have the required ecosystem packages installed before executing the notebook pipeline:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

Open week3-ashish-kumar-ipyn.ipynb using Jupyter Notebook, Lab, or Kaggle.

Run all code cells sequentially to visualize the elbow plot, clustering profiles, classification metrics, and feature importance distributions.
```


# 📌 Week 4 Assignment: Deep Learning Performance Analysis on CIFAR-10

## 🎯 Project AIM
* Build an image classification model on CIFAR-10.
* Build an image classification model on CIFAR-10 and analyze performance across architectures and training strategies using both Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN).

---

## 🚀 Project Overview
This repository contains the complete submission for the Week 4 Assignment, focusing on benchmarking image classification pipelines on the CIFAR-10 dataset (60,000 $32 \times 32$ color images across 10 distinct classes). 

The core objective is to analyze how different network topologies interpret 2D spatial pixel data and to empirically evaluate the shift from feature extraction to severe model overfitting under extended training periods.

---

## 🔬 The 8-Stage Experimental Workflow

The project systematically benchmarks model performance across 8 distinct sequential configurations:

### 🔹 Phase 1: The ANN Baseline Limitations
* **Experiment 0: Base ANN (10 Epochs)** Built using a flattened input vector layout. Suffered from heavy underfitting due to the absolute destruction of 2D spatial pixel correlations, setting the project's performance floor.
* **Experiment 1: ANN + Extra Hidden Layer** Added depth via an additional Dense layer. The structural performance bottleneck remained unchanged, proving that increasing raw capacity in an ANN cannot solve its inability to extract localized spatial features.

### 🔹 Phase 2: The CNN Paradigm Shift
* **Experiment 2: Base CNN (10 Epochs)** Introduced Convolutional layers, feature maps, and Max Pooling. Preserving spatial dependencies natively caused a massive accuracy surge over the ANN baseline.
* **Experiment 3: CNN with Expanded Filter Capacity (64-128-256)** Inflated channel dimensions to evaluate the network's ability to capture complex, high-level geometric patterns (edges, shapes, orientations).
* **Experiment 4: CNN Extended Training (20 Epochs)** **The Overfitting Threshold.** Running the unregularized network for 20 epochs caused training accuracy to jump to 94.8% while validation performance completely stagnated and validation loss exploded.

### 🔹 Phase 3: Advanced Regularization & Defenses
* **Experiment 5: CNN + Early Stopping Callback** Implemented dynamic monitoring on validation loss (`patience=3`) to automatically halt training prior to model divergence, protecting generalizability.
* **Experiment 6: CNN + Live Data Augmentation** Introduced random runtime geometric transformations (Horizontal Flips, Rotations, Zooms) to prevent pixel location memorization.
* **Experiment 7: Improved CNN (The Ultimate Regularized Network)** An all-in-one defensive architecture combining an advanced deep CNN with **Batch Normalization** after every block, **L2 Weight Regularization** to penalize large weights, and **Heavy Dropout (0.5)** to enforce feature redundancy.

---

## 📊 Final Performance Benchmarking

| Stage | Model Architecture / Strategy | Test Accuracy | Generalization Status |
| :---: | :--- | :---: | :--- |
| **0** | Base ANN (10 Epochs) | `0.4280` (42.80%) | Underfitting (Incapable of capturing spatial patterns) |
| **1** | ANN + Extra Hidden Layer | `0.4317` (43.17%) | Structurally bottlenecked by flattening image arrays |
| **2** | Base CNN (10 Epochs) | `0.7249` (72.49%) | Strong feature extraction, but highly prone to overfitting |
| **3** | CNN (64-128-256 Filters) | `0.7262` (72.62%) | Higher filter capacity, but plateaus without regularization |
| **4** | CNN (Extended to 20 Epochs) | `0.7286` (72.86%) | **Severe Overfitting** (Train Acc: 94.8% vs Test: 72%) |
| **5** | CNN + EarlyStopping | `0.7005` (70.05%) | Successfully halts training prior to heavy divergence |
| **6** | CNN + Data Augmentation | `0.6953` (69.53%) | High generalization, but limited by shallow network size |
| **7** | 🏆 **Improved CNN (Final)** | **`0.7064` (70.64%)** | **Best & Most Robust Model** (Zero overfitting, stable loss curves) |

---

## 🎯 Key Architectural Takeaways

### 1. Spatial Retention is Paramount
Flattening multidimensional image arrays strips away neighboring pixel contexts. CNNs use shared-weight filters across local receptive fields to natively map coordinate hierarchies, making them fundamentally superior to traditional feed-forward networks for vision tasks.

### 2. High Training Metrics Can Be Deceptive
A high training score (such as the **94.82%** observed in Experiment 4) is frequently an illusion of success. Without optimization safety rails, the network becomes a simple look-up table, making it fail completely on unseen real-world test sets.

### 3. The Generalization Victory
Our final **Improved CNN** model represents the most robust system. By enforcing simultaneous mathematical constraints (L2 penalty stabilizes weights, Dropout introduces redundancy, and Augmentation forces spatial abstraction), the training loss ($1.32$) and validation loss ($1.32$) converged in perfect harmony, making it ready for reliable deployment on unseen datasets.

## 📐 Project Pipeline & Architecture Workflow
The repository maps a systematic approach divided into three progressive execution blocks:

Dataset (CIFAR-10) ➔ 1. Flat Vectors ➔ Linear Dense Layers ➔ [ANN Baseline]
                     ➔ 2. 2D Tensor Grid ➔ Conv2D + Pooling Layers ➔ [CNN Capacity Shift]
                     ➔ 3. Augmentation ➔ BatchNorm ➔ L2 + Dropout ➔ [Improved CNN Final]

---

## 🚀 Dependencies & Installation
Ensure you have the required ecosystem packages installed before executing the notebook pipeline:

```bash
pip install tensorflow numpy pandas matplotlib








