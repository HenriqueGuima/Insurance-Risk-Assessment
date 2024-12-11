#!/usr/bin/env python
# coding: utf-8

# # Dataset of Insurance
# 
# # Objective
# 
# Identify high-risk customers who are more likely to make claims or commit fraud.
# 
# # Model Type
# Classification
# 
# # Target Variable
# Create a derived target variable (e.g., High Risk = 1, Low Risk = 0) based on factors such as frequent claims or unusual patterns in policy usage.
# 
# # Features
# Include Previous Claims, Credit Score, Vehicle Age, Policy Type, Health Score, etc.

# # Exploratory Data Analysis

# In[87]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# In[88]:


# Load the data
data = pd.read_csv('insurance_premium_dataset.csv')

# Display the first 5 rows of the data
print(data.head())


# In[89]:


# Display the data info
print(data.info())


# In[90]:


# Display unique values of marital_status
print(data['Marital Status'].unique())


# In[91]:


# Display Education level unique values
print(data['Education Level'].unique())


# In[92]:


# Display the unique values of the occupation
print(data['Occupation'].unique())


# In[93]:


# Display the unique values of the Health Score
print(data['Health Score'].unique())


# In[94]:


# Display the unique values of the Policy Type
print(data['Policy Type'].unique())


# In[95]:


# Display the unique values of the Location
print(data['Location'].unique())


# In[96]:


# Display the unique values of the Previous Claims
print(data['Previous Claims'].unique())


# In[97]:


# Display the unique values of the Premium Amount
print(data['Customer Feedback'].unique())


# In[98]:


# Display the unique values of the Smoking Status
print(data['Smoking Status'].unique())


# In[99]:


# Display the unique values of the Exercise Frequency
print(data['Exercise Frequency'].unique())


# In[100]:


# Display the unique values of the Property Type
print(data['Property Type'].unique())


# Age to Int - v
# Convert gender to binary - v
# Convert marital status to 0, 1, 2 - v
# Convert number of dependents to int - v
# Convert Education level to 0, 1, 2, 3
# Convert Occupation to 0, 1, 2
# Convert Policy Type to 0, 1, 2
# Convert Location to 0, 1, 2
# Convert Previous Claims to int
# Credit Score to int
# Premium Amount to int
# Customer Feedback to int
# Smoking Status to binary
# Exercise Frequency to int
# Property Type to int

# # Missing values

# In[101]:


# Check for missing values

sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

print(data.isnull().sum())


# In[102]:


# Drop missing values
new_data = data.dropna()
print(new_data.head())


# # Data Transform

# In[103]:


print(new_data.describe())


# ## Gender
#     Male - 0
#     Female - 1
# 
# ## Marital Status
#     Single - 0
#     Married - 1
#     Divorced - 2
# 
# ## Education Level 
#     Master's - 0
#     Bachelor's - 1
#     PhD - 2
#     High Schoold - 3
# 
# ## Occupation
#     Self Employed - 0
#     Employed - 1
#     Unemployed - 2
# 
# ## Policy Type
#     Comprehensive - 0
#     Premium - 1
#     Basic - 2
# 
# ## Location 
#     Urban - 0
#     Suburban - 1
#     Rural - 2
# 
# ## Property Type
#     Apartment - 0
#     House - 1
#     Condo - 2
# 
# ## Customer Feedback
#     Poor - 0
#     Average - 1
#     Good - 2
# 
# ## Exercise Frequency
#     Daily - 0
#     Weekly - 1
#     Monthly - 2
#     Rarely - 3
# 
# ## Smoking Status
#     Yes - 1
#     No - 0
# 
# 

# In[104]:


# Convert categorical features to numerical
new_data['Gender'] = new_data['Gender'].map({'Female': 1, 'Male': 0})
new_data['Marital Status'] = new_data['Marital Status'].map({'Single': 0, 'Married': 1, 'Divorced': 2})
new_data['Education Level'] = new_data['Education Level'].map({'Master\'s': 0, 'Bachelor\'s': 1, 'PhD': 2, 'High School': 3})
new_data['Occupation'] = new_data['Occupation'].map({'Self-Employed': 0, 'Employed': 1, 'Unemployed': 2})
new_data['Policy Type'] = new_data['Policy Type'].map({'Comprehensive': 0, 'Premium': 1, 'Basic': 2})
new_data['Location'] = new_data['Location'].map({'Urban': 0, 'Suburban': 1, 'Rural': 2})
new_data['Property Type'] = new_data['Property Type'].map({'Apartment': 0, 'House': 1, 'Condo': 2})
new_data['Customer Feedback'] = new_data['Customer Feedback'].map({'Poor': 0, 'Average': 1, 'Good': 2})
new_data['Exercise Frequency'] = new_data['Exercise Frequency'].map({'Daily': 0, 'Weekly': 1, 'Monthly': 2, 'Rarely': 3})
new_data['Smoking Status'] = new_data['Smoking Status'].map({'Yes': 1, 'No': 0})


# In[105]:


# Change data types
# new_data.loc[:, 'Age'] = new_data['Age'].astype(int)

# Age to int
new_data['Age'] = new_data['Age'].astype(int)

# new_data.loc[:, 'Gender'] = new_data['Gender'].astype(bool)
new_data['Number of Dependents'] = new_data['Number of Dependents'].astype(int)
new_data['Previous Claims'] = new_data['Previous Claims'].astype(int)
new_data['Credit Score'] = new_data['Credit Score'].astype(int)
new_data['Premium Amount'] = new_data['Premium Amount'].astype(int)
new_data['Smoking Status'] = new_data['Smoking Status'].astype(int)
new_data['Property Type'] = new_data['Property Type'].astype(int)

# Display the first 5 rows of the new data
# print(new_data.head())
print(new_data.info())


# SAVE DATA

# In[106]:


# Save the cleaned data
new_data.to_csv('cleaned_insurance_data.csv', index=False)

exec(open('eda_ml_unsup_learning_insurance.py').read())

