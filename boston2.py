#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import sklearn.datasets
from sklearn import model_selection


# In[2]:


from sklearn.datasets import load_boston
boston = load_boston()


# In[3]:


print(boston.keys())


# In[4]:


print(boston.data.shape)


# In[5]:


print(boston.feature_names)


# In[6]:


print(boston.DESCR)


# In[7]:


bost = pd.DataFrame(boston.data)
print(bost.head())


# In[8]:


bost['PRICE'] = boston.target
print(bost.head())


# In[9]:


bost=(bost-bost.mean())/(bost.max()-bost.min())


# In[10]:


print(bost.describe())


# In[11]:


X = bost.drop('PRICE', axis = 1)
Y = bost['PRICE']


# In[24]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


lr = LinearRegression()


# In[29]:


lr.fit(X_train,Y_train)


# In[30]:


pred = lr.predict(X_test)


# In[32]:


plt.scatter(Y_test,pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Predicted value-true value=Linear Regression")

plt.show()


# In[33]:


from sklearn.metrics import mean_squared_error


# In[34]:


mse = mean_squared_error(Y_test,pred) 
print("Mean Square Error : ", mse)


# In[35]:


#Building the model
m=0
c=0
L = 0.0001 
it= 1000  
n = float(len(X))
#Building Gradient Descent 
for i in range(it): 
    Y_pred = m*X + c
    D_m = (-2/n) * sum(X * (Y - Y_pred))  
    D_c = (-2/n) * sum(Y - Y_pred) 
    m = m - L * D_m  
    c = c - L * D_c  
    print (m, c)


# In[37]:


pred


# In[38]:


plt.title('Cost Function J', size = 30)
plt.xlabel('No. of iterations', size=20)
plt.ylabel('Cost', size=20)
plt.show()


# In[39]:


from sklearn.linear_model import Ridge


# In[40]:


from sklearn.linear_model import Lasso


# In[41]:


from sklearn.preprocessing import StandardScaler


# In[42]:


from sklearn.preprocessing import PolynomialFeatures


# In[43]:


from sklearn.pipeline import Pipeline


# In[44]:


steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=0.0001, normalize=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, Y_train)

print('Training Score: {}'.format(ridge_pipe.score(X_train, Y_train)))
print('Test Score: {}'.format(ridge_pipe.score(X_test, Y_test)))


# In[45]:


steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Lasso(alpha=3, normalize=True))
]

lasso_pipe = Pipeline(steps)

lasso_pipe.fit(X_train, Y_train)

print('Training score: {}'.format(lasso_pipe.score(X_train,Y_train)))
print('Test score: {}'.format(lasso_pipe.score(X_test, Y_test)))


# In[ ]:




