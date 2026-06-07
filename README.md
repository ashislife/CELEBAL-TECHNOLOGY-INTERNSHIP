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

---

## 🚀 Dependencies & How to Run
Make sure you have the following packages installed before running the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib

Open the file week2-ashish-kumar-ipynb.ipynb in Jupyter Notebook, Lab, or Kaggle.

Run all cells sequentially to recreate the data analysis, visualizations, and model training.
