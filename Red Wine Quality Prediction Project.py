#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
wine_sample=pd.read_csv("winequality-red.csv")


# In[2]:


wine_sample.head()


# In[3]:


wine_sample.isnull().sum()


# In[4]:


wine_sample.describe()


# In[5]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine_sample)


# In[6]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine_sample)


# In[7]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine_sample)


# In[8]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'chlorides', data = wine_sample)


# In[9]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine_sample)


# In[10]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine_sample)


# In[11]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'density', data = wine_sample)


# In[12]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'pH', data = wine_sample)


# In[13]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'sulphates', data = wine_sample)


# In[14]:


fig = plt.figure(figsize = (5,5))
sns.barplot(x = 'quality', y = 'alcohol', data = wine_sample)


# In[ ]:


correlation = wine_sample.corr()


# In[22]:


# constructing a heatmap to understand the correlation b/t the colums
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, fmt='if', annot=True, annot_kws={'size':8},cmap='RdPu')


# In[23]:


# sepret data and lable
x = wine_sample.drop('quality' ,axis=1)
print(x)


# In[24]:


y = wine_sample['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
print(y)


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=3)
print(y.shape, y_train.shape, y_test.shape)


# In[26]:


#Random Forest Classifier
model = RandomForestClassifier()
model.fit(x_train, y_train)


# In[27]:


#accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print('accuracy : ', test_data_accuracy)


# In[29]:


#building a predctive system
input_data = (7.3,0.65,0.0,1.2,0.065,15.0,21.0,0.9946,3.39,0.47,10.0)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==1):
    print('GOOD QUALITY WINE')
else:
    print('BAD QUALITY WINE')


# # GOOD QUALITY WINE
