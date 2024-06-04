#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

glass1 = pd.read_csv(r"C:\PythonDatasets\Noise_borderline_dataset\Datasets\04clover5z-800-7-0-BI.csv")

glass1.head()


# In[2]:


#glass1['Class'] = glass1['Class'].map({'negative':0, 'positive':1,})
#glass1['Class'].replace({'negative': 0, 'positive': 1})


# In[3]:


glass1.iloc[:,-1:]


# In[4]:


import pandas as pd
label_counts = glass1['Class'].value_counts()
print(label_counts)


# In[5]:


print(1, round(glass1['Class'].value_counts()[1]/len(glass1) * 100,2), '% of the dataset')
print(0, round(glass1['Class'].value_counts()[0]/len(glass1) * 100,2), '% of the dataset')


# In[6]:


glass1.isnull().sum().sum()


# In[7]:


nan_df = glass1.isna()
nan_count_per_column = nan_df.sum()
print(nan_count_per_column)


# In[8]:


#Breaking into X and Y
X = glass1.iloc[:, 0:-1]
y = glass1.iloc[:, -1]


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[10]:


print("Train dataset：",x_train.shape)
print("Train dataset labels：",y_train.shape)
print("Test dataset：",x_test.shape)
print("Test dataset labels：",y_test.shape)


# In[11]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
# Now you can use X_train_scaled, X_test_scaled, y_train, y_test for model training and evaluation


# In[12]:


'''
#RESAMPLING OF DATA USING SMOTE
from collections import Counter
print("Before SMOTE:", Counter(y_train))
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

# Perform SMOTE oversampling
X_resampled, y_resampled = smote.fit_resample(X_train_scaled,y_train)

# Display class distribution after oversampling
print("After SMOTE:", Counter(y_resampled))
'''


# In[13]:


#################################### ROS ###############################################
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
print("Before ROS:", Counter(y_train))
ros = RandomOverSampler(random_state=42)

# Perform random oversampling
X_resampled, y_resampled = ros.fit_resample(X_train_scaled, y_train)

# Print the number of instances in each class after oversampling
print("After ROS:", Counter(y_resampled))


# In[19]:


#Combing the data for train and test into single variable for measureing complexity
# Combine the train and test data into a single variable
import numpy as np
X_combined = np.vstack((X_resampled, X_test_scaled))
y_combined = np.hstack((y_resampled, y_test))

# Check the shapes of the combined data
print("Combined X shape:", X_combined.shape)
print("Combined y shape:", y_combined.shape)


# In[20]:


#COMBINING X AND Y INTO A SINGLE DATAFRAME
X_df = pd.DataFrame(X_combined, columns=[f"feature{i+1}" for i in range(X.shape[1])])

# Create a Series for the target variable
y_series = pd.Series(y_combined, name='Class')

# Combine X and y into a single DataFrame
df = pd.concat([X_df, y_series], axis=1)

# Print the first few rows of the combined DataFrame
print(df.head())


# In[21]:


import pandas as pd
label_counts = df['Class'].value_counts()
print(label_counts)


# In[17]:


'''
#SAVE AS A CSV FILE
import csv
df.to_csv("04clover5z-800-7-0-BI(ROS).csv")
'''


# In[22]:


import problexity as px


# In[23]:


# Initialize CoplexityCalculator with default parametrization
cc = px.ComplexityCalculator()

# Fit model with data
cc.fit(X_combined,y_combined)
#arr = np.add(arr, image.flatten(), out=arr, casting="unsafe")


# In[24]:


cc.score()


# In[25]:


cc.report()


# In[26]:


# Import matplotlib
import matplotlib.pyplot as plt

# Prepare figure
fig = plt.figure(figsize=(7,7))

# Generate plot describing the dataset
cc.plot(fig, (1,1,1))


# In[23]:


from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


# In[24]:


svm_clf = SVC()
rf_clf = RandomForestClassifier(random_state=42)
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier(max_iter = 2000,random_state = 40)
nb_clf = GaussianNB()


# In[25]:


classifiers = [svm_clf, rf_clf, knn_clf, mlp_clf,nb_clf]
for clf in classifiers:
    clf.fit(X_resampled, y_resampled)
    y_pred = clf.predict(X_test_scaled) 
    accuracy = accuracy_score(y_test, y_pred)
    f1= f1_score(y_test,y_pred)
    print(f"{clf.__class__.__name__} F1 Score: {f1:.2f}")
       


# In[ ]:





# In[ ]:





# In[ ]:




