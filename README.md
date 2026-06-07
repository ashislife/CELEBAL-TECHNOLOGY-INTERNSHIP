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
