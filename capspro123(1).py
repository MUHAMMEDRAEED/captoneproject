#!/usr/bin/env python
# coding: utf-8

# # Title: Cancer Grade Prediction using Machine Learning
# 
# ## Author: MUHAMMED RAEED MK
# 
# ## Date : 01/07/2024
# 
# 
# 
# # Table of Contents
# 
#     1. Title Page
#     2. Table of Contents
#     3. Overview of Problem Statement
#     4. Objective
#     5. Data Loading and Preprocessing
#     6. Exploratory Data Analysis (EDA)
#     7. Feature Engineering
#     8. Model Training and Evaluation
#     9. Hyperparameter Tuning
#     10. Results and Conclusion
#     11. Model Deployment
# 
# 
# # Overview of Problem Statement
# 
# 
# The problem of cancer grade prediction is a critical issue in the field of cancer research.
# Cancer grading is a process of determining the severity of cancer based on various factors such as tumor size, lymph node involvement, and metastasis.
# Accurate cancer grade prediction can aid in diagnosis, treatment planning, and patient prognosis.
# However, cancer grade prediction is a complex task due to the involvement of multiple factors and the lack of a clear understanding of the underlying mechanisms.
# 
# 
# # Objective
# 
# 
# The objective of this project is to develop a machine learning model that accurately predicts cancer grades based on various features, including mutation statuses and patient information.
# The expected outcomes of this project are:
#      To develop a model that accurately predicts cancer grades with high accuracy and low error rates.
#      To identify the most important features that contribute to cancer grade prediction.
#      To provide insights into the relationships between mutation statuses and cancer grades.
# 

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import re  # Python regular expressions (regex) library
import joblib


# In[3]:


df = pd.read_csv("TCGA_GBM_LGG_Mutations_all.csv")
df.head()


# In[4]:


df.shape, df.columns


# In[5]:


import re # Python regular expressions(regex) library

def convertAgeToNumber(age_in_words):
    """converts given age in string format into equivalent float value"""
    
    # regex to find numbers('\d+') in the string that have boundaries('\b') on both sides
    numbers = re.findall(r'\b\d+\b', age_in_words) 
    numerical_age = None # Handle invalid strings
    if len(numbers) >= 1: # Handle the years specified
        numerical_age = float(numbers[0])
    if len(numbers) >= 2: # Handle the days specified
        numerical_age += float(numbers[1])/365
    return numerical_age    

df['Age_at_diagnosis'] = df['Age_at_diagnosis'].map(convertAgeToNumber)
df.Age_at_diagnosis[0]

# Fill missing values with mode

modes = df.mode().iloc[0] 

df.fillna(modes, inplace=True)


# Process Grade column

df['Grade'] = np.where(df['Grade'] == 'LGG', 0 , 1)


# Define categorical and continuous columns

categorical=['Gender', 'Race', 'IDH1', 

      'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 

      'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 

      'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']

continuos=['Age_at_diagnosis']

dependent="Grade"


# One-hot encode categorical columns

df_onehot = pd.get_dummies(df, columns=categorical)


# Print the resulting dataframe

print(df_onehot.head())


# In[6]:


df


# In[7]:


df.isna().sum() 


# In[8]:


modes = df.mode().iloc[0] # calculate mode of each column
df.fillna(modes, inplace=True) # fill NaN in each column with corresponding mode.


# In[9]:


df.isna().sum() 


# In[10]:


# Next we process the independent variable, 
# since our model expects numbers as input during training, we use the where() method in numpy 
# to change the values in the Grade column to integers


# In[11]:


# np.where(condition, true-val, false-val)
df['Grade'] = np.where(df['Grade'] == 'LGG', 0 , 1)


# In[12]:


df


# In[13]:


# Removed Project, Case_ID & Primary_Diagnosis columns
categorical=['Gender', 'Race', 'IDH1', 
      'TP53', 'ATRX', 'PTEN', 'EGFR', 'CIC', 'MUC16', 'PIK3CA', 'NF1', 
      'PIK3R1', 'FUBP1', 'RB1', 'NOTCH1', 'BCOR', 'CSMD3', 'SMARCA4', 
      'GRIN2A', 'IDH2', 'FAT4', 'PDGFRA']
continuos=['Age_at_diagnosis']
dependent="Grade"


# In[14]:


for cat in categorical:
    df[cat] = pd.Categorical(df[cat])


# In[15]:


df[categorical] = df[categorical].apply(lambda x: x.cat.codes)
# check to make sure that all categorical variable that are being considered 
# for model training have been replaced with corresponding code numbers.
df.head()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the style for seaborn plots
sns.set(style="whitegrid")


# In[17]:


# Distribution of the Grade in the info dataset
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(x='Grade', data=df)
plt.title('Distribution of Cancer Grades')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()


# In[18]:


# Age distribution based on Grade
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age_at_diagnosis', hue='Grade', kde=True, multiple="stack")
plt.title('Age Distribution by Cancer Grade')
plt.xlabel('Age at Diagnosis')
plt.ylabel('Count')
plt.show()


# In[19]:


# Relationship between mutation statuses and Grade
plt.figure(figsize=(10, 6))
sns.countplot(x='IDH1', hue='Grade', data=df)
plt.title('IDH1 Mutation Status by Grade')
plt.xlabel('IDH1 Mutation Status')
plt.ylabel('Count')
plt.show()


# In[20]:


plt.figure(figsize=(10, 6))
sns.countplot(x='TP53', hue='Grade', data=df)
plt.title('TP53 Mutation Status by Grade')
plt.xlabel('TP53 Mutation Status')
plt.ylabel('Count')
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.countplot(x='ATRX', hue='Grade', data=df)
plt.title('ATRX Mutation Status by Grade')
plt.xlabel('ATRX Mutation Status')
plt.ylabel('Count')
plt.show()


# In[22]:


plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[23]:


from sklearn.model_selection import train_test_split

np.random.seed(42) 
train_df,valid_df = train_test_split(df, test_size=0.25)


# In[24]:


def extract_data_labels(df):
    data = df[categorical+continuos].copy()
    label = df[dependent]
    return data, label

train_data,train_label = extract_data_labels(train_df)
valid_data,valid_label = extract_data_labels(valid_df)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define the model
# The model has 100 trees, leaf nodes in each tree will have a minimum of 5 samples
rf = RandomForestClassifier(100, min_samples_leaf=5) 

# Train the model
rf.fit(train_data, train_label)

# Evaluate the results
print(f'Accuracy: {accuracy_score(valid_label, rf.predict(valid_data)): .4f} ')


# In[26]:


from sklearn.metrics import classification_report
print(classification_report(y_true=valid_label, y_pred=rf.predict(X=valid_data)))


# In[27]:


models = {
    'Logistic Regression': LogisticRegression(max_iter=300),
    'SVC': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=5),
    'MLP Classifier': MLPClassifier(max_iter=300)
}


# In[28]:


for model_name, model in models.items():
    model.fit(train_data, train_label)
    predictions = model.predict(valid_data)
    accuracy = accuracy_score(valid_label, predictions)
    print(f'{model_name} Accuracy: {accuracy:.4f}')
    print(classification_report(valid_label, predictions))


# In[29]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Hyperparameter Tuning (Optional): Example for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_data, train_label)

# Displaying the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)


# In[30]:


# Save trained model
filename = 'Grid_gbr.joblib'
joblib.dump(grid_search, filename)
print("Model saved successfully.")

# Load saved model
loaded_model = joblib.load(filename)
print("Model loaded successfully.")

