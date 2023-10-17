#!/usr/bin/env python
# coding: utf-8

#  <center> 
#     Housing Price Detection Using Liner Regression <br>
#     Sumitted by Muni Aswanth Prasad A
# </center>

# In[ ]:





# ### Importing necessary libraries

# In[62]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Import the Dataset and Pre-Processing the Data

# In[64]:


data = pd.read_csv('Housing.csv')


# In[65]:


data.shape


# In[66]:


data.head()


# In[67]:


data.info()


# In[68]:


data.describe()


# In[69]:


data.isnull().any()


# In[70]:


data.isnull().sum()


# In[71]:


data.columns


# In[78]:


plt.figure(figsize=(50,5))
sns.countplot(data)


# In[79]:


g = sns.PairGrid(data)
g.map_upper(sns.scatterplot,color='Purple')
g.map_lower(sns.scatterplot, color='Black')
g.map_diag(plt.hist,color='#0146B6')


# In[80]:


sns.histplot(data['Price'],color="purple")


# In[81]:


corr=data.corr()
corr


# In[82]:


plt.subplots(figsize=(10,10))
sns.heatmap(corr,annot=True)


# ## Taining the model

# In[83]:


X = data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms','Avg. Area Number of Bedrooms', 'Area Population']]
y = data['Price']


# #  Train and Test Split

# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Training Model

# In[86]:


from sklearn.linear_model import LinearRegression


# In[87]:


lm = LinearRegression()


# In[88]:


lm.fit(X_train,y_train)


# ## Model Evaluation 

# In[89]:


print(lm.intercept_)


# In[90]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[91]:


predictions = lm.predict(X_test)


# In[92]:


plt.scatter(y_test,predictions,color="purple")


# In[93]:


sns.displot((y_test-predictions),bins=50,color="purple");


# In[94]:


from sklearn import metrics


# In[95]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:





# In[ ]:





# In[ ]:




