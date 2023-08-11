#!/usr/bin/env python
# coding: utf-8

# # TASK: Email Spam Detection With Machine Learning

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


data = pd.read_csv(r'C:\Users\parveen\Desktop\spam.csv', encoding= 'ISO-8859-1')


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.shape


# In[7]:


data.size


# In[8]:


data.info()


# In[10]:


data.describe()


# In[11]:


data.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace = True)


# In[12]:


data.head()


# In[13]:


data = data.rename(columns = {'v1': 'Target', 'v2':'Message'})


# In[14]:


data.isnull().sum()


# In[15]:


data.duplicated().sum()


# In[16]:


data.drop_duplicates(keep='first', inplace =True)


# In[17]:


data.duplicated().sum()


# In[18]:


data.size


# In[19]:


from sklearn.preprocessing import LabelEncoder
encoder= LabelEncoder()
data['Target']=encoder.fit_transform(data['Target'])
data['Target']


# In[20]:


data.head()


# In[21]:


plt.pie(data['Target'].value_counts(), labels= ['ham','spam'], autopct = "%0.2f")
plt.show()


# In[22]:


x = data['Message']


# In[24]:


print(x) 


# In[27]:


y = data['Target']


# In[28]:


print(y)


# In[29]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size= 0.2, random_state = 3)


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


# In[31]:


cv = CountVectorizer()


# In[32]:


x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)


# In[33]:


print(x_train_cv)


# In[34]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[35]:


lr.fit(x_train_cv,y_train)
prediction_train = lr.predict(x_train_cv)


# In[36]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,prediction_train)*100)


# In[37]:


prediction_test = lr.predict(x_test_cv)


# In[38]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction_test)*100)


# In[ ]:
