#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import Dependencies
get_ipython().run_line_magic('matplotlib', 'inline')

# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')


# In[5]:


train = pd.read_csv(r"C:\Users\KALPANA K\Downloads\train.csv")
test = pd.read_csv(r"C:\Users\KALPANA K\Downloads\test.csv")
gender_submission = pd.read_csv(r"C:\Users\KALPANA K\Downloads\Titanic.csv")


# In[6]:


train.head()


# In[7]:


train.Age.plot.hist()


# In[8]:


test.head()


# In[9]:


gender_submission.head()


# In[11]:


train.isnull().sum()


# In[12]:


df_bin = pd.DataFrame() 
df_con = pd.DataFrame()


# In[13]:


train.head()


# In[14]:


fig = plt.figure(figsize=(20,1))
sns.countplot(y='Survived', data=train);
print(train.Survived.value_counts())


# In[15]:


df_bin['Survived'] = train['Survived']
df_con['Survived'] = train['Survived']


# In[16]:


sns.distplot(train.Pclass)


# In[17]:


train.Name.value_counts()


# In[18]:


plt.figure(figsize=(20, 5))
sns.countplot(y="Sex", data=train);


# In[19]:


df_bin['Sex'] = train['Sex']
df_bin['Sex'] = np.where(df_bin['Sex'] == 'female', 1, 0) 

df_con['Sex'] = train['Sex']


# In[20]:


fig = plt.figure(figsize=(10, 10))
sns.distplot(df_bin.loc[df_bin['Survived'] == 1]['Sex'], kde_kws={'bw':1.5, 'label': 'Survived'});
sns.distplot(df_bin.loc[df_bin['Survived'] == 0]['Sex'], kde_kws={'bw':1.5, 'label': 'Did not survive'});


# In[21]:


sns.countplot(y="Ticket", data=train);


# In[23]:


sns.countplot(y='Embarked', data=train);


# In[32]:


df_bin['Embarked'] = train['Embarked']
df_con['Embarked'] = train['Embarked']


# In[33]:


print(len(df_con))
df_con = df_con.dropna(subset=['Embarked'])
df_bin = df_bin.dropna(subset=['Embarked'])
print(len(df_con))


# In[34]:


one_hot_cols = df_bin.columns.tolist()
one_hot_cols.remove('Survived')
df_bin_enc = pd.get_dummies(df_bin, columns=one_hot_cols)

df_bin_enc.head()


# In[37]:


df_embarked_one_hot = pd.get_dummies(df_con['Embarked'], 
                                     prefix='embarked')

df_sex_one_hot = pd.get_dummies(df_con['Sex'], 
                                prefix='sex')


# In[39]:


df_con_enc = pd.concat([df_con, 
                        df_embarked_one_hot, 
                        df_sex_one_hot], axis=1)

# Drop the original categorical columns (because now they've been one hot encoded)
df_con_enc = df_con_enc.drop(['Sex', 'Embarked'], axis=1)


# In[41]:


df_con_enc.head(20)


# In[42]:


selected_df = df_con_enc
selected_df.head()


# In[43]:


X_train = selected_df.drop('Survived', axis=1) # data
y_train = selected_df.Survived 


# In[44]:


def fit_ml_algo(algo, X_train, y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv


# In[45]:


start_time = time.time()
train_pred_log, acc_log, acc_cv_log = fit_ml_algo(LogisticRegression(), 
                                                               X_train, 
                                                               y_train, 
                                                                    10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))


# In[ ]:




