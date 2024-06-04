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


# In[5]:


import pandas as pd
label_counts = glass1['Class'].value_counts()
print(label_counts)


# In[6]:


print(1, round(glass1['Class'].value_counts()[1]/len(glass1) * 100,2), '% of the dataset')
print(0, round(glass1['Class'].value_counts()[0]/len(glass1) * 100,2), '% of the dataset')


# In[7]:


glass1.isnull().sum().sum()


# In[8]:


nan_df = glass1.isna()
nan_count_per_column = nan_df.sum()
print(nan_count_per_column)


# In[9]:


#Breaking into X and Y
import numpy
X = glass1.iloc[:, 0:-1]
y = glass1.iloc[:, -1]


# In[10]:


import problexity as px


# In[11]:


# Initialize CoplexityCalculator with default parametrization
cc = px.ComplexityCalculator(float)
# Fit model with data
cc.fit(X,y)
#arr = np.add(arr, image.flatten(), out=arr, casting="unsafe")



# In[12]:


cc.score()


# In[13]:


cc.report()


# In[15]:


# Import matplotlib
import matplotlib.pyplot as plt

# Prepare figure
fig = plt.figure(figsize=(7,7))

# Generate plot describing the dataset
cc.plot(fig, (1,1,1))


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


print("Train dataset：",x_train.shape)
print("Train dataset labels：",y_train.shape)
print("Test dataset：",x_test.shape)
print("Test dataset labels：",y_test.shape)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)
# Now you can use X_train_scaled, X_test_scaled, y_train, y_test for model training and evaluation


# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB


# In[ ]:


svm_clf = SVC()
rf_clf = RandomForestClassifier(random_state=42)
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier(max_iter = 2000,random_state = 40)
nb_clf = BernoulliNB()


# In[ ]:


classifiers = [svm_clf, rf_clf, knn_clf, mlp_clf,nb_clf]
for clf in classifiers:
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1= f1_score(y_test,y_pred)
    print(f"{clf.__class__.__name__} F1 Score: {f1:.2f}")
       

