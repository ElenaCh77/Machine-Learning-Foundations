#!/usr/bin/env python
# coding: utf-8

# # Lab 2: ML Life Cycle: Data Understanding and Data Preparation

# In[31]:


import os
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sns


# In this lab, you will practice the second and third steps of the machine learning life cycle: data understanding and data preparation. You will beging preparing your data so that it can be used to train a machine learning model that solves a regression problem. Note that by the end of the lab, your data set won't be completely ready for the modeling phase, but you will gain experience using some common data preparation techniques. 
# 
# You will complete the following tasks to transform your data:
# 
# 1. Build your data matrix and define your ML problem:
#     * Load the Airbnb "listings" data set into a DataFrame and inspect the data
#     * Define the label and convert the label's data type to one that is more suitable for modeling
#     * Identify features
# 2. Clean your data:
#     * Handle outliers by building a new regression label column by winsorizing outliers
#     * Handle missing data by replacing all missing values in the dataset with means
# 3. Perform feature transformation using one-hot encoding
# 4. Explore your data:
#     * Identify two features with the highest correlation with label
#     * Build appropriate bivariate plots to visualize the correlations between features and the label
# 5. Analysis:
#     * Analyze the relationship between the features and the label
#     * Brainstorm what else needs to be done to fully prepare the data for modeling

# ## Part 1. Build Your Data Matrix (DataFrame) and Define Your ML Problem

# #### Load a Data Set and Save it as a Pandas DataFrame

# We will be working with the Airbnb NYC "listings" data set. Use the specified path and name of the file to load the data. Save it as a Pandas DataFrame called `df`.

# In[32]:


# Do not remove or edit the line below:
filename = os.path.join(os.getcwd(), "data", "airbnbData.csv")


# **Task**: Load the data and save it to DataFrame `df`.
# 
# <i>Note:</i> You may receive a warning message. Ignore this warning.

# In[33]:


# YOUR CODE HERE
df = pd.read_csv(filename, low_memory=False)


# ####  Inspect the Data
# 

# <b>Task</b>: Display the shape of `df` -- that is, the number of rows and columns.

# In[34]:


# YOUR CODE HERE
print(df.shape)


# <b>Task</b>: Display the column names.

# In[35]:


# YOUR CODE HERE
print(list(df.columns))


# **Task**: Get a peek at the data by displaying the first few rows, as you usually do.

# In[36]:


# YOUR CODE HERE
print(df.head())


# #### Define the Label

# Assume that your goal is to train a machine learning model that predicts the price of an Airbnb. This is an example of supervised learning and is a regression problem. In our dataset, our label will be the `price` column. Let's inspect the values in the `price` column.

# In[37]:


df['price']


# Notice the `price` column contains values that are listed as $<$currency_name$>$$<$numeric_value$>$. 
# <br>For example, it contains values that look like this: `$120`. <br>
# 
# **Task**:  Obtain the data type of the values in this column:

# In[38]:


# YOUR CODE HERE
print(df['price'].dtype)


# Notice that the data type is "object," which in Pandas translates to the String data type.

# **Task**:  Display the first 15 unique values of  the `price` column:

# In[39]:


# YOUR CODE HERE
print(df['price'].unique()[:15])


# In order for us to use the prices for modeling, we will have to transform the values in the `price` column from strings to floats. We will:
# 
# * remove the dollar signs (in this case, the platform forces the currency to be the USD, so we do not need to worry about targeting, say, the Japanese Yen sign, nor about converting the values into USD). 
# * remove the commas from all values that are in the thousands or above: for example, `$2,500`. 
# 
# The code cell below accomplishes this.

# In[40]:


df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].astype(float)


# **Task**:  Display the first 15 unique values of  the `price` column again to make sure they have been transformed.

# In[41]:


# YOUR CODE HERE
print(df['price'].unique()[:15])


# #### Identify Features

# Simply by inspecting the data, let's identify some columns that should not serve as features - those that will not help us solve our predictive ML problem. 

# Some that stand out are columns that contain website addresses (URLs).
# 
# **Task**: Create a list which contains the names of columns that contain URLs. Save the resulting list to variable `url_colnames`.
# 
# *Tip*: There are different ways to accomplish this, including using Python list comprehensions.

# In[42]:


url_colnames = [col for col in df.columns if 'url' in col.lower()]
# YOUR CODE HERE
url_colnames


# **Task**: Drop the columns with the specified names contained in list `url_colnames` in place (that is, make sure this change applies to the original DataFrame `df`, instead of creating a temporary new DataFrame object with fewer columns).

# In[49]:


# YOUR CODE HERE
df.drop(columns=url_colnames, inplace = True, errors = 'ignore')


# **Task**: Display the shape of the data to verify that the new number of columns is what you expected.

# In[50]:


# YOUR CODE HERE
print(df.shape)


# **Task**: In the code cell below, display the features that we will use to solve our ML problem.

# In[51]:


# YOUR CODE HERE
print(list(df.columns))


# **Task**: Are there any other features that you think may not be well suited for our machine learning problem? Note your findings in the markdown cell below.

# Some columns, like id, host_id, name, description, host_about, and date columns such as last_scraped or first_review, are not useful for predicting price. They are either just identifiers, free text, or dates that don't directly help our model, so I would not use them as features.

# ## Part 2. Clean Your Data
# 
# Let's now handle outliers and missing data.

# ### a. Handle Outliers
# 
# Let us prepare the data in our label column. Namely, we will detect and replace outliers in the data using winsorization.

# **Task**: Create a new version of the `price` column, named `label_price`, in which you will replace the top and bottom 1% outlier values with the corresponding percentile value. Add this new column to the DataFrame `df`.

# Remember, you will first need to load the `stats` module from the `scipy` package:

# In[54]:


# YOUR CODE HERE 
from scipy import stats
bottom_price = np.percentile(df['price'],1)
top_price = np.percentile(df['price'],99)

df['label_price'] = np.clip(df['price'], bottom_price, top_price)


# Let's verify that the new column `label_price` was added to DataFrame `df`:

# In[55]:


df.head()


# **Task**: Check that the values of `price` and `label_price` are *not* identical. 
# 
# You will do this by subtracting the two columns and finding the resulting *unique values*  of the resulting difference. <br>Note: If all values are identical, the difference would not contain unique values. If this is the case, outlier removal did not work.

# In[56]:


# YOUR CODE HERE
unique_differences = (df['label_price'] - df['price']).unique()

print(unique_differences)


# ### b. Handle Missing Data
# 
# Next we are going to find missing values in our entire dataset and impute the missing values by
# replace them with means.

# #### Identifying missingness

# **Task**: Check if a given value in the data is missing, and sum up the resulting values by columns. Save this sum to variable `nan_count`. Print the results.

# In[57]:


nan_count = df.isna().sum() # YOUR CODE HERE
nan_count


# Those are more columns than we can eyeball! For this exercise, we don't care about the number of missing values -- we just want to get a list of columns that have *any* missing values.
# 
# <b>Task</b>: From the variable `nan_count`, create a new series called `nan_detected` that contains `True` or `False` values that indicate whether the number of missing values is *not zero*:

# In[63]:


nan_detected = nan_count != 0 # YOUR CODE HERE
nan_detected


# Since replacing the missing values with the mean only makes sense for the columns that contain numerical values (and not for strings), let us create another condition: the *type* of the column must be `int` or `float`.

# **Task**: Create a series that contains `True` if the type of the column is either `int64` or `float64`. Save the results to the variable `is_int_or_float`.

# In[64]:


is_int_or_float = df.dtypes.isin(['int64','float64']) # YOUR CODE HERE
is_int_or_float


# <b>Task</b>: Combine the two binary series (`nan_detected` and `is_int_or_float`) into a new series named `to_impute`. It will contain the value `True` if a column contains missing values *and* is of type 'int' or 'float'

# In[65]:


to_impute = nan_detected & is_int_or_float # YOUR CODE HERE
to_impute


# Finally, let's display a list that contains just the selected column names contained in `to_impute`:

# In[68]:


df.columns[to_impute]


# We just identified and displayed the list of candidate columns for potentially replacing missing values with the column mean.

# Assume that you have decided that you should impute the values for these specific columns: `host_listings_count`, `host_total_listings_count`, `bathrooms`, `bedrooms`, and `beds`:

# In[69]:


to_impute_selected = ['host_listings_count', 'host_total_listings_count', 'bathrooms',
       'bedrooms', 'beds']


# #### Keeping record of the missingness: creating dummy variables 

# As a first step, you will now create dummy variables indicating the missingness of the values.

# **Task**: For every column listed in `to_impute_selected`, create a new corresponding column called `<original-column-name>_na`. These columns should contain the a `True`or `False` value in place of `NaN`.

# In[72]:


# YOUR CODE HERE 
for col in to_impute_selected:
    df[col + '_na'] = df[col].isna()


# Check that the DataFrame contains the new variables:

# In[73]:


df.head()


# #### Replacing the missing values with mean values of the column

# **Task**: For every column listed in `to_impute_selected`, fill the missing values with the corresponding mean of all values in the column (do not create new columns).

# In[74]:


# YOUR CODE HERE
for col in to_impute_selected:
    df[col].fillna(df[col].mean(), inplace=True)


# Check your results below. The code displays the count of missing values for each of the selected columns. 

# In[75]:


for colname in to_impute_selected:
    print("{} missing values count :{}".format(colname, np.sum(df[colname].isnull(), axis = 0)))


# Why did the `bathrooms` column retain missing values after our imputation?

# **Task**: List the unique values of the `bathrooms` column.

# In[76]:


# YOUR CODE HERE
list(df['bathrooms'].unique())


# The column did not contain a single value (except the `NaN` indicator) to begin with.

# ## Part 3. Perform One-Hot Encoding

# Machine learning algorithms operate on numerical inputs. Therefore, we have to transform text data into some form of numerical representation to prepare our data for the model training phase. Some features that contain text data are categorical. Others are not. For example, we removed all of the features that contained URLs. These features were not categorical, but rather contained what is called unstructured text. However, not all features that contain unstructured text should be removed, as they can contain useful information for our machine learning problem. Unstructured text data is usually handled by Natural Language Processing (NLP) techniques. You will learn more about NLP later in this course. 
# 
# However, for features that contain categorical values, one-hot encoding is a common feature engineering technique that transforms them into binary representations. 

# We will first choose one feature column to one-hot encode: `host_response_time`. Let's inspect the unique values this feature can have. 

# In[77]:


df['host_response_time'].unique()


# Note that each entry can contain one of five possible values. 
# 
# **Task**: Since one of these values is `NaN`, replace every entry in the column `host_response_time` that contains a `NaN` value with the string 'unavailable'.

# In[78]:


# YOUR CODE HERE
df['host_response_time'].fillna('unavailable', inplace=True)


# Let's inspect the `host_response_time` column to see the new values.

# In[79]:


df['host_response_time'].unique()


# **Task**: Use `pd.get_dummies()` to one-hot encode the `host_response_time` column. Save the result to DataFrame `df_host_response_time`. 

# In[80]:


df_host_response_time = pd.get_dummies(df['host_response_time']) 
# YOUR CODE HERE
df_host_response_time


# **Task**: Since the `pd.get_dummies()` function returned a new DataFrame rather than making the changes to the original DataFrame `df`, add the new DataFrame `df_host_response_time` to DataFrame `df`, and delete the original `host_response_time` column from DataFrame `df`.
# 

# In[81]:


# YOUR CODE HERE
df = pd.concat([df, df_host_response_time], axis=1)
df.drop('host_response_time', axis=1, inplace=True)


# Let's inspect DataFrame `df` to see the changes that have been made.

# In[82]:


df.columns


# #### One-hot encode additional features
# 
# **Task**: Use the code cell below to find columns that contain string values  (the 'object' data type) and inspect the *number* of unique values each column has.

# In[83]:


# YOUR CODE HERE
object_cols = df.select_dtypes(include='object')
print(object_cols.nunique())


# **Task**: Based on your findings, identify features that you think should be transformed using one-hot encoding.
# 
# 1. Use the code cell below to inspect the unique *values* that each of these features have.

# In[112]:


# YOUR CODE HERE
cols_to_encode = [
    'neighbourhood_group_cleansed',
    'room_type',
    'host_is_superhost',
    'host_has_profile_pic',
    'host_identity_verified',
    'has_availability',
    'instant_bookable'
]

for col in cols_to_encode:
    if col in df.columns:
        print(f"{col}: {df[col].unique()}")
        print('-' * 40)
    else:
        print(f"{col} not found in DataFrame.")
        print('-' * 40)
        
        print(df.columns.tolist())


# 2.  List these features and explain why they would be suitable for one-hot encoding. Note your findings in the markdown cell below.

# room_type is suitable for one-hot encoding because it is a categorical feature with a small number of possible values like Entire home/apt, Private room, Hotel room, and Shared room. One-hot encoding transforms each room type into its own binary column, allowing machine learning algorithms to process this information as numerical input.
# 
# neighbourhood_group_cleansed. This feature represents the borough of each listing such as Manhattan, Brooklyn, Queens, Bronx, and Staten Island. Since there are only a few possible values and they do not have any meaningful order, one-hot encoding is a good way to convert this categorical data into a format suitable for machine learning models

# **Task**: In the code cell below, one-hot encode one of the features you have identified and replace the original column in DataFrame `df` with the new one-hot encoded columns. 

# In[131]:


# YOUR CODE HERE
print(df.columns)
df = pd.concat([df, pd.get_dummies(df['bathrooms_text'], prefix='bathrooms_text')], axis=1)
df.drop('bathrooms_text', axis=1, inplace=True)


# ## Part 4. Explore Your Data

# You will now perform exploratory data analysis in preparation for selecting your features as part of feature engineering. 
# 
# #### Identify Correlations
# 
# We will focus on identifying which features in the data have the highest correlation with the label.

# Let's first run the `corr()` method on DataFrame `df` and save the result to the variable `corr_matrix`. Let's round the resulting correlations to five decimal places:

# In[132]:


corr_matrix = round(df.corr(),5)
corr_matrix


# The result is a computed *correlation matrix*. The values on the diagonal are all equal to 1 because they represent the correlations between each column with itself. The matrix is symmetrical with respect to the diagonal.<br>
# 
# We only need to observe correlations of all features with the column `label_price` (as opposed to every possible pairwise correlation). Se let's query the `label_price` column of this matrix:
# 
# **Task**: Extract the `label_price` column of the correlation matrix and save the results to the variable `corrs`.

# In[133]:


corrs = corr_matrix['label_price'] # YOUR CODE HERE
corrs


# **Task**: Sort the values of the series we just obtained in the descending order and save the results to the variable `corrs_sorted`.

# In[136]:


corrs_sorted = corrs.sort_values(ascending=False) # YOUR CODE HERE
corrs_sorted


# **Task**: Use Pandas indexing to extract the column names for the top two correlation values and save the results to the Python list `top_two_corr`. Add the feature names to the list in the order in which they appear in the output above. <br> 
# 
# <b>Note</b>: Do not count the correlation of `label` column with itself, nor the `price` column -- which is the `label` column prior to outlier removal.

# In[137]:


top_two_corr = top_two_corr = [col for col in corrs_sorted.index if col not in ['label_price', 'price']][:2] 
# YOUR CODE HERE
top_two_corr


# #### Bivariate Plotting: Produce Plots for the Label and Its Top Correlates
# 
# Let us visualize our data.

# We will use the `pairplot()` function in `seaborn` to plot the relationships between the two features and the label.

# **Task**: Create a DataFrame `df_corrs` that contains only three columns from DataFrame `df`: the label, and the two columns which correlate with it the most.

# In[138]:


df_corrs = df[['label_price'] + top_two_corr] # YOUR CODE HERE
df_corrs


# **Task**: Create a `seaborn` pairplot of the data subset you just created. Specify the *kernel density estimator* as the kind of the plot, and make sure that you don't plot redundant plots.
# 
# <i>Note</i>: It will take a few minutes to run and produce a plot.

# In[141]:


# YOUR CODE HERE
sns.pairplot(df_corrs, kind="kde", corner=True)
plt.show()


# ## Part 5: Analysis
# 
# 1. Think about the possible interpretation of the plot. Recall that the label is the listing price. <br> How would you explain the relationship between the label and the two features? Is there a slight tilt to the points cluster, as the price goes up?<br>
# 2. Are the top two correlated features strongly or weakly correlated with the label? Are they features that should be used for our predictive machine learning problem?
# 3. Inspect your data matrix. It has a few features that contain unstructured text, meaning text data that is neither numerical nor categorical. List some features that contain unstructured text that you think are valuable for our predictive machine learning problem. Are there other remaining features that you think need to be prepared for the modeling phase? Do you have any suggestions on how to prepare these features?
# 
# Record your findings in the cell below.

# 1)Looking at my pairplot, I can see that when the number of bedrooms or how many people the place can accommodate goes up, the listing price usually goes up too. The points aren’t in a perfect straight line, but there’s definitely a pattern showing that bigger places cost more. There are some listings that are way more expensive than the rest, but overall, the trend makes sense.
# 2)There are some columns in the data like name, description, neighborhood_overview, and amenities that have a lot of text. I think these could be helpful because hosts might mention special features, cool stuff about the neighborhood, or extra amenities that could make a listing more valuable.
# 3) Some columns still need a bit of cleaning, like turning categories into numbers (one-hot encoding) and filling in any missing values. For the text features, I’d probably use some basic NLP stuff like picking out keywords or even checking if the description sounds positive or not. Handling outliers in price could also help so that the model doesn’t get thrown off by super expensive listings.
# 
# 
