#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install necessary packages
get_ipython().run_line_magic('pip', 'install -q     numpy==1.26.4     pandas==2.2.2     matplotlib==3.9.0     seaborn==0.13.2     scikit-learn==1.4.2     xlrd==2.0.1')


# In[1]:


# The following code cell accomplishes two tasks: it imports the required 
#libraries and loads a YouTube video. This video provides a fundamental understanding of how random forests operate.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import warnings

# Suppress warnings:
def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.filterwarnings('ignore')

sns.set_context('notebook')
sns.set_style('white')

# The following code loads the YouTube video:
from IPython.display import YouTubeVideo
YouTubeVideo('gkXX4h3qYm4', width=800, height=452)


# In[ ]:


# The task for this tutorial is to predict the probability of a customer defaulting on a loan.
# The target variable is "default payment." 
#Details of the data measurements can be found in [UCI's data repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients?utm_source=ibm_developer&utm_content=in_content_link&utm_id=tutorials_awb-random-forest-predict-credit-defaults).

# This step might take a couple of minutes to complete. Please be patient.


# In[6]:


# Import the data set
df = pd.read_excel('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UEym8G6lwphKjuhkLgkXAg/default%20of%20credit%20card%20clients.xls', header=1)


# In[7]:


# Explore the first five rows of the data set
df.head(5)


# In[ ]:


#Each row represents an individual. Some columns that could be useful are:

# Credit limit (LIMIT_BAL)
# Prior payment status (PAY_0,...,PAY_6)
# Bill and payment amounts (BILL_AMT, PAY_AMT)
# Target variable that indicates default next month (default payment next month)
# The following code renames the default column. The ID column is also dropped because it doesn't contain any information relevant to the analysis.


# In[8]:


# Rename the columns
df.rename({'default payment next month': 'DEFAULT'}, axis='columns', inplace=True)

#Remove the ID column as it is not informative
df.drop('ID', axis=1, inplace=True)
df.head()


# In[9]:


# Analyze missing data
# One key step is to check for null values or other invalid input that will cause the model to throw an error.

# Check dimensions for invalid values
print(f"SEX values include: {df['SEX'].unique()}")
print(f"MARRIAGE values include: {df['MARRIAGE'].unique()}")
print(f"EDUCATION values include: {df['EDUCATION'].unique()}")

# Count missing or null values
print(f"Number of missing values in SEX: {len(df[pd.isnull(df.SEX)])}")
print(f"Number of missing values in MARRIAGE: {len(df[pd.isnull(df.MARRIAGE)])}")
print(f"Number of missing values in EDUCATION: {len(df[pd.isnull(df.EDUCATION)])}")
print(f"Number of missing values in AGE: {len(df[pd.isnull(df.AGE)])}")

# Count of invalid data in EDUCATION and MARRIAGE
invalid_count = len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)])
print(f"Number of invalid data points in EDUCATION or MARRIAGE: {invalid_count}")


# In[ ]:


### The output indicates that some of the data does not align with the data definitions, specifically the EDUCATION and MARRIAGE columns.

# EDUCATION includes three types of invalid values, which are 0, 5, and 6.
# MARRIAGE includes 0 as an invalid value.
# Assume that a 0 encoding is supposed to represent missing data and that a value of 5 or 6 within EDUCATION is representative of other unspecified education levels (for example, Ph.D. or a master's degree), which is not represented within the data definition. 68 rows exist in the DataFrame where either the EDUCATION or the MARRIAGE column is zero.

# Next, let's filter the rows where the EDUCATION and MARRIAGE columns have non-zero values.

# The following code creates a new DataFrame with the missing values for EDUCATION and MARRIAGE removed. We end up with 29,932 rows remaining.


# In[10]:


print(f"shape of data: {df.shape}")

#Filter the DataFrame
df_no_missing_data = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]
print(f"shape of no_missing_data: {df_no_missing_data.shape}")


# In[ ]:


# The next step is to check if the target variable, which indicates whether someone defaulted, is balanced. 
# The chart shows counts of people who have defaulted (1) and haven't defaulted (0).
# Unsurprisingly, most people have not defaulted on their loans.

# To address this class imbalance, you must down sample the data.


# In[11]:


# Explore distribution of data set
# count plot on ouput variable
ax = sns.countplot(x = df_no_missing_data['DEFAULT'], palette = 'rocket')

#add data labels
# ax.bar_label(ax.containers[0])
for container in ax.containers:
    ax.bar_label(container)

# add plot title
plt.title("Observations by Classification Type")

# show plot
plt.show()


# In[ ]:


# Downsample the data set
# The first step in downsampling is to split the data based on those who defaulted on their loan and those who did not default on their loan.

# You will randomly select 1,000 samples from each category.

# The two data sets are then merged back together to create an analysis data set.


# In[12]:


# split data
df_no_default = df_no_missing_data.loc[(df_no_missing_data['DEFAULT']==0)]
df_default = df_no_missing_data.loc[(df_no_missing_data['DEFAULT']==1)]

# downsample the data set
df_no_default_downsampled = resample(df_no_default, replace=False, n_samples=1000, random_state=0)
df_default_downsampled = resample(df_default, replace=False, n_samples=1000, random_state=0)

#check ouput
print(f"Length of df_no_default_downsampled: {len(df_no_default_downsampled)}")
print(f"Length of df_default_downsampled: {len(df_default_downsampled)}")

# merge the data sets
df_downsample = pd.concat([df_no_default_downsampled, df_default_downsampled ])
print(f"Shape of df_downsample: {df_downsample.shape}")


# In[ ]:


# Hot encode the independent variables
# Next, you convert each category into a binary variable with a value of 0 or 1.
# Pandas has a very convenient function to do just this, called get_dummies.

# Why hot encode?

# Improved model performance
# Avoid implicit ordering
# Compatibility and consistency
# One thing to keep in mind when creating models is to avoid bias.
# One very important way to do this is to not use variables associated with protected attributes as independent variables. 
# In this case, SEX, AGE, and MARRIAGE clearly fall into that category.
# EDUCATION is somewhat more ambiguous. Because it is not critical for the purposes of this tutorial, this is dropped as well.


# In[13]:


# isolate independent variables
X = df_downsample.drop(['DEFAULT','SEX', 'EDUCATION', 'MARRIAGE','AGE'], axis=1).copy()
print(f"Shape of X: {X.shape}")

# NOTE: 'PAY_1' is not shown in original data
X_encoded = pd.get_dummies(data=X, columns=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'])
print(f"Shape of X_encoded: {X_encoded.shape}")
X_encoded.head()


# In[ ]:


# Split the data set
# Splitting the data into test and training sets is critical for understanding how your model performs on new data. 
# The random forest model uses the training data set to learn what factors should become decision nodes.
# The test set helps you evaluate how often those decisions lead to the correct decision.


# In[14]:


# Split the data
y = df_downsample['DEFAULT'].copy()
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=0)

print(f'X_train.shape: {X_train.shape}')
print(f'X_test.shape: {X_test.shape}')
print(f'y_train.shape: {y_train.shape}')
print(f'y_test.shape: {y_test.shape}')


# In[ ]:


# Classify accounts and evaluate the model
# Now, it's time to build an initial random forest model
# by fitting it by using the training data and evaluating the resulting model using the test data.
#To make that evaluation easier, you plot the results using a confusion matrix.


# In[15]:


# apply RandomForestClassifier
clf_rf = RandomForestClassifier(max_depth=2, random_state=0)
clf_rf.fit(X_train, y_train)

#calculate overall accuracy
y_pred = clf_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')

class_names = ['Did Not Default', 'Defaulted']

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the percentage of correctly predicted instances for each class
for i, class_name in enumerate(class_names):
    correct_predictions = cm[i, i]
    total_predictions = cm[i, :].sum()
    class_accuracy = correct_predictions / total_predictions * 100
    print(f'Percentage of correctly predicted {class_name}: {class_accuracy:.2f}%')

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()


# In[ ]:


# As you can see, the model performance could be improved.


# Optimize the model with hyperparameter tuning
# Cross validation and GridSearchCV are useful tools for finding better hyperparameters for models.
# When it comes to random forest models, you'll focus on max_depth, min_samples_split, min_samples_leaf.
# Here's a quick overview of what those hyperparameters mean:

# max_depth: The maximum number levels the decision trees that make up the random forest are allowed to have
# min_samples_split: The minimum number of samples that must be in a node for a decision split to be allowed
# min_samples_leaf: The minimum number of samples that must exist in a leaf node
# Another commonly used hyperparameter is max_features. 
# This is the number of features that the model will try out when attempting to create a decision node. 
# The n_estimators hyperparameter controls the number of decision trees that are created as part of the random forest model. 
# For more details and other hyperparameters that can be tuned, see the sklearn random forest documentation.


# In[16]:


param_grid = {
    'max_depth':[3,4,5],
    'min_samples_split':[3,4,5],
    'min_samples_leaf':[3,4,5],
}

rf_random = RandomizedSearchCV(
    estimator=clf_rf, 
    param_distributions=param_grid, 
    n_iter=27, 
    cv=3, 
    random_state=0, 
    verbose=1,
    n_jobs = -1,
)

# Fit the random search model
rf_random.fit(X_train, y_train)

# Output the best hyperparameters found
best_params = rf_random.best_params_
print(f'Best parameters found: {best_params}')
print(f'Best estimator is: {rf_random.best_estimator_}')

# Refit the model using the best hyperparameters
best_clf_rf = rf_random.best_estimator_

# In case you want to check all parameters currently in use
# print(f'Parameters currently in use: {best_clf_rf.get_params()}')

# Train the refitted model
best_clf_rf.fit(X_train, y_train)

# Calculate overall accuracy
y_pred = best_clf_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')

# Plot the confusion matrix
class_names = ['Did Not Default', 'Defaulted']
disp = ConfusionMatrixDisplay.from_estimator(
    best_clf_rf,
    X_test,
    y_test,
    display_labels=class_names,
    cmap=plt.cm.Blues,
)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




