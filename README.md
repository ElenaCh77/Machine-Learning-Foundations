# Machine-Learning-Foundations
Labs by Cornell Tech (Break Through Tech AI Program). The course focused on building ethical, real-world ML models using Python, covering the full ML lifecycle, from data preprocessing to model evaluation. These labs include hands-on work with real datasets, exploring core ML concepts such as regression, classification, and bias evaluation.


## 🧠 Lab 1: ML Life Cycle – Business Understanding & Problem Formulation

In this lab, I practiced the first step of the **machine learning life cycle**:  
**formulating a machine learning problem** and understanding the business context.

Before defining the problem, I also reviewed core Python tools used throughout the ML workflow.

### ✅ What I Did
- Worked with **NumPy arrays** and mathematical operations  
- Created and analyzed **Pandas DataFrames**  
- Explored and summarized data using NumPy and Pandas functions  
- Visualized patterns in the data using **Matplotlib**  
- Practiced connecting data exploration to ML problem formulation  

### 🛠 Tools & Libraries
`Python` · `NumPy` · `Pandas` · `Matplotlib`


## 🧠 Lab 2: ML Life Cycle – Data Understanding & Data Preparation

In this lab, I practiced the second and third phases of the **machine learning life cycle**:  
**data understanding** and **data preparation**.  

The goal was to explore the Airbnb “Listings” dataset, define the regression problem, and begin transforming the data so it can be used to train a machine-learning model.

### ✅ What I Did
- Loaded and inspected the Airbnb dataset  
- Defined the **regression label** and converted it to the proper data type  
- Identified relevant **features** for the model  
- Cleaned the data by:
  - **Winsorizing outliers** to create a new regression label  
  - Replacing missing values with **mean imputation**  
  - Applying **one-hot encoding** to categorical variables  
- Explored feature relationships:
  - Found the two features most correlated with the label  
  - Built **bivariate plots** to visualize correlations  
- Analyzed how features relate to the target and identified additional work needed before modeling  

### 🛠 Tools & Libraries
`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn`


## 🧠 Lab 3: ML Life Cycle – Modeling

In this lab, I practiced the **modeling phase** of the machine learning life cycle by training and comparing two classification models: **Decision Trees** and **K-Nearest Neighbors (KNN)**.  
The goal was to experiment with different hyperparameters and determine which model performs best on the Airbnb “Listings” dataset.

### ✅ What I Did
- Built the DataFrame, defined the **classification label**, and selected features  
- Prepared the data by:
  - Applying **one-hot encoding** to categorical variables  
  - Creating labeled examples  
  - Splitting into **training** and **testing** sets  
- Trained multiple **Decision Tree classifiers** with varying `max_depth` values  
  - Evaluated accuracy and plotted **max depth vs. accuracy**  
- Trained several **KNN classifiers** with different `k` values  
  - Compared model accuracy and plotted **k vs. accuracy**  
- Analyzed which model performed best and explored how hyperparameters and data characteristics influence performance  

### 🛠 Tools & Libraries
`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

## 🧠 Lab 4: ML Life Cycle – Modeling (Logistic Regression From Scratch)

In this lab, I continued working in the **modeling phase** of the machine learning life cycle by **building a logistic regression classifier from scratch**.  
The goal was to implement the mathematical components behind logistic regression — including the log loss, gradient, Hessian, and gradient descent — without relying on scikit-learn’s internal algorithms.

### ✅ What I Did
- Loaded and prepared the Airbnb “Listings” dataset  
- Defined the **classification label** and selected features  
- Created labeled examples for training  
- Designed a full **LogisticRegressionScratch** Python class that can:
  - Compute predicted probabilities  
  - Compute the **gradient** of the log loss  
  - Compute the **Hessian**  
  - Update weights using **gradient descent**  
  - Check convergence based on tolerance or max iterations  
  - Fit a logistic regression model end-to-end  
- Trained the custom model and compared it against **scikit-learn’s LogisticRegression** to benchmark performance  

### 🛠 Tools & Libraries
`Python` · `NumPy` · `Pandas` · `scikit-learn`

## 🧠 Lab 5: ML Life Cycle – Evaluation & Deployment

In this lab, I continued practicing the **evaluation** and early **deployment** phases of the machine learning life cycle.  
The focus was on **model selection**, **hyperparameter tuning**, and preparing a logistic regression model for future use.

### ✅ What I Did
- Loaded and prepared the Airbnb “Listings” dataset  
- Defined the **classification label** and selected features  
- Created labeled examples and split the data into training and testing sets  
- Trained and evaluated a **baseline Logistic Regression** model using scikit-learn defaults  
- Performed **GridSearchCV** to identify the optimal regularization hyperparameter  
- Trained and evaluated the optimized model  
- Plotted and compared:
  - **Precision–Recall curves**  
  - **ROC curves** and computed **AUC**  
- Performed **feature selection** to improve interpretability and performance  
- Saved (serialized) the final model to make it **persistent for future deployment**

### 🛠 Tools & Libraries
`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

## 🧠 Lab 6: Train Various Regression Models & Compare Their Performances

In this lab, I focused on the **modeling and evaluation** phases of the machine learning life cycle by training multiple regression models and comparing their performance on the Airbnb “Listings” dataset. The goal was to explore individual regressors, ensemble methods, and model stacking.

### ✅ What I Did
- Loaded the dataset, defined the **regression label**, and selected features  
- Created labeled examples and split the data into **training** and **test** sets  
- Trained, tested, and evaluated two **individual regressors**  
- Used **stacking ensemble methods** to combine the regressors  
- Trained and evaluated **Gradient Boosting Regressors**  
- Trained and evaluated **Random Forest Regressors**  
- Compared all models using metrics such as **MSE** and **R²**  
- Visualized and analyzed model performance across all approaches  

### 🛠 Tools & Libraries
`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

## 🧠 Lab 7: Implementing a Convolutional Neural Network Using Keras

In this lab, I explored the **modeling** and **evaluation** phases of the ML life cycle by building a **Convolutional Neural Network (CNN)** using Keras.  
The goal was to classify hand-written digits — a classic computer vision problem — using the MNIST dataset.

### ✅ What I Did
- Defined the **classification label** and feature set  
- Loaded and split the MNIST dataset into training and test sets  
- Inspected and visualized the images to understand their structure  
- Preprocessed the data so it could be used in a neural network  
- Built a multi-layer **Convolutional Neural Network** using Keras  
- Trained the CNN and monitored its performance  
- Evaluated the final model on both training and test data  
- Used the provided demo (“Implementing a Neural Network Using Keras”) as a reference

### 🛠 Tools & Libraries
`Python` · `TensorFlow / Keras` · `NumPy` · `Matplotlib` · `Seaborn`

## 🧠 Lab 8: Define and Solve an ML Problem of Your Choosing

In this final lab, I applied the complete **machine learning life cycle** to a project of my choice.  
I selected the **censusData.csv** dataset and defined a predictive problem based on the structure and characteristics of the data.

### ✅ What I Did
- Loaded the **censusData.csv** dataset into a DataFrame  
- Defined the **ML problem**, identifying the label (target) and selecting relevant features  
- Performed **exploratory data analysis (EDA)** to understand distributions, correlations, and key patterns  
- Created a structured **project plan** for data preparation, modeling, and evaluation  
- Prepared the data using appropriate preprocessing methods (cleaning, encoding, transformations)  
- Trained and evaluated a machine-learning model tailored to the census dataset  
- Iteratively improved the model based on evaluation metrics and insights from the data  

### 🛠 Tools & Libraries
`Python` · `pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn`

