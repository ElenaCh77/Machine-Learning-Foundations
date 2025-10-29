# Machine-Learning-Foundations
Labs by Cornell Tech (Break Through Tech AI Program). The course focused on building ethical, real-world ML models using Python, covering the full ML lifecycle, from data preprocessing to model evaluation. These labs include hands-on work with real datasets, exploring core ML concepts such as regression, classification, and bias evaluation.

```
Lab 1 – Airbnb Listings Data
```
In this lab, I worked with the Airbnb “Listings” dataset, which contains New York City Airbnb data from December 2021 (modified for the Break Through Tech AI program).
The objective was to explore, clean, and analyze the dataset to identify patterns and prepare the data for future machine learning tasks.

```
Lab 2 –  ML Life Cycle: Data Understanding and Data Preparation
```
In this lab, I practiced the second and third steps of the machine learning life cycle: data understanding and data preparation. I prepared a data so that it can be used to train a machine learning model that solves a regression problem.
I completed the following tasks to transform a data:

1. I build]t a data matrix and define ML problem:
 - Loaded the Airbnb "listings" data set into a DataFrame and inspect the data
 - Defined the label and convert the label's data type to one that is more suitable for modeling
 - I have identified features
2. Clean a data:
 - Handle outliers by building a new regression label column by winsorizing outliers
 - Handled missing data by replacing all missing values in the dataset with means
 - Performed feature transformation using one-hot encoding
3. Explore a data:
 - I have identified two features with the highest correlation with label
 - Built appropriate bivariate plots to visualize the correlations between features and the label
4. Analysis:
 - Analyzed the relationship between the features and the label
```
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


