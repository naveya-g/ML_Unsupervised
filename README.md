# 🚗🔍 ML Unsupervised Learning - Car Segmentation & Vehicle Silhouette Classification

This project applies unsupervised and supervised machine learning techniques to two automobile datasets - grouping cars by fuel efficiency patterns using clustering, and classifying vehicle types from their silhouette features using dimensionality reduction and SVM.

---

### Project Objective

To discover hidden patterns in car data without using predefined labels - grouping similar cars together based on their attributes - and to classify vehicle silhouettes into distinct types by reducing the number of features to only the most informative ones, making the model leaner and faster without sacrificing accuracy.

---

### Dataset Summary

**Part 1 - Car Fuel Segmentation**
- Two files merged on index: `Car name.csv` and `Car-Attributes.json`
- Key features: MPG (miles per gallon), Cylinders, Engine Displacement, Horsepower, Weight, Model Year, Origin
- Key challenges: `hp` column stored as object type with `?` as missing values (replaced with NaN, imputed with median); most cars from the US (origin = 1), years ranging 1972–1980

**Part 2 - Vehicle Silhouette Classification**
- Single file: `vehicle.csv` with 18 numeric geometric features extracted from vehicle silhouettes
- 4 vehicle types used: Double Decker Bus, Chevrolet Van, Saab 9000, Opel Manta 400 - consolidated into **3 classes**: Bus, Van, Car
- Key challenges: missing values imputed with column-wise median; cars harder to distinguish from each other than from bus/van

---

### Analysis Workflow

**Part 1 - Car Segmentation (K-Means Clustering)**
- Replaced `?` in `hp` with NaN and imputed with median; standardized all features using Z-score scaling before clustering
- Applied K-Means for K=2 to 10; used both **WSS (Within Sum of Squares)** and **Average Distortion** methods with `KneeLocator` to automatically detect the elbow point - both methods pointed to **K=5**
- Validated with **Silhouette Score** - K=5 (score higher) outperformed K=3, confirming 5 as the optimal number of clusters
- Visualized clusters using **PCA-reduced 2D scatter plot** and added `cluster_value5` as a new feature to the dataset

**Part 2 - Vehicle Silhouette Classification (PCA + SVM)**
- Imputed missing values with column-wise median; applied Z-score scaling; built a **baseline SVM** (kernel=RBF) without PCA as reference
- Applied PCA incrementally - selected **minimum components explaining ≥ 90% variance** (10 components out of 18), reducing feature space significantly
- Trained SVM on PCA components; then improved with **manual hyperparameter tuning** (C, kernel, gamma) and **GridSearchCV** with 5-fold cross-validation
- Compared all 4 model variants (SVM, SVM+PCA, SVM+PCA+Tune, SVM+PCA+Grid) on train and test accuracy to identify the best-performing configuration

---

### Model Performance - Part 2 (Vehicle Classification)

| Model | Train Accuracy | Test Accuracy | Key Observation |
|---|---|---|---|
| SVM (no PCA) | Highest | Highest | Strong baseline - best on raw features |
| SVM + PCA | 81% | ~71.7% | PCA reduced accuracy; fewer features lost information |
| SVM + PCA + Tune | ~93% | Lower | Overfit on training data |
| **SVM + PCA + GridSearch** ✅ | **~89.5%** | **~81%** | **Best balance - strong generalization with PCA** |

> SVM without PCA had the highest raw accuracy, but **SVM + PCA + GridSearch** offered the best balance between dimensionality reduction and performance, making it the recommended model for scalable deployment.

---

### Tools Used

- Python, NumPy, Pandas, Matplotlib, Seaborn
- Clustering: `KMeans`, `silhouette_score`, `KneeLocator`, `cdist` (scipy)
- Dimensionality Reduction: `PCA` (Scikit-learn)
- Classification: `SVC` with RBF, Linear, and Poly kernels
- Hyperparameter Tuning: Manual grid loop + `GridSearchCV` with 5-fold cross-validation
- Preprocessing: Z-score scaling (`zscore`), `StandardScaler`
- Environment: Google Colab

---

### 🔍 Use Case

**Part 1** helps automotive analysts and manufacturers segment their vehicle lineup by fuel efficiency and engine characteristics - enabling targeted marketing, pricing strategies, and product positioning across different customer segments.

**Part 2** enables automated vehicle type identification from camera silhouettes - useful in traffic monitoring systems, parking management, and smart city infrastructure where vehicles need to be classified in real time without manual intervention.
