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
Lab 3: Machine Learning Life Cycle – Modeling
```
In this lab, I practiced the modeling phase of the ML life cycle using the Airbnb listings dataset. I trained and compared Decision Tree and K-Nearest Neighbors (KNN) models to solve a classification problem.

✅ What I Did

Loaded and explored the dataset with pandas and numpy
Defined the label and selected features
Performed one-hot encoding and split data into train/test sets
Trained multiple Decision Tree models with different max_depth values and plotted their accuracy
Trained KNN models with various k values and compared results
Determined the best-performing model and analyzed key factors affecting performance

🧩 Tools & Libraries
Python, pandas, numpy, matplotlib, seaborn, scikit-learn
```


