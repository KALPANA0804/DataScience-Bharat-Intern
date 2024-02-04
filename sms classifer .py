#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline    
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap


# In[7]:


file_path = "C:\\Users\\KALPANA K\\Downloads\\spam (1).csv"
data = pd.read_csv(file_path, encoding='latin-1')


# In[9]:


data.info()


# In[10]:


data.head()


# In[11]:


to_drop = ["Unnamed: 2","Unnamed: 3","Unnamed: 4"]
data = data.drop(data[to_drop], axis=1)

data.rename(columns = {"v1":"Target", "v2":"Text"}, inplace = True)
data.head()


# In[14]:


plt.figure(figsize=(12,8))
fg = sns.countplot(x= data["Target"], palette= ('green','red'))
fg.set_title("Count Plot of Classes", color="#58508d")
fg.set_xlabel("Classes", color="#58508d")
fg.set_ylabel("Number of Data points", color="#58508d")


# In[17]:


import nltk


nltk.download('punkt')


data["No_of_Characters"] = data["Text"].apply(len)
data["No_of_Words"] = data.apply(lambda row: nltk.word_tokenize(row["Text"]), axis=1).apply(len)
data["No_of_Sentences"] = data.apply(lambda row: nltk.sent_tokenize(row["Text"]), axis=1).apply(len)


statistics = data.describe().T
print(statistics)


# In[18]:


plt.figure(figsize=(12,8))
fg = sns.pairplot(data=data, hue="Target",palette=("cyan","yellow"))
plt.show(fg)


# In[25]:


plt.figure(figsize=(8, 8))
explode = (0.1, 0)  # explode the 1st slice (optional, for emphasis)
colors = ['#70a288', '#e35f5f']  # custom colors
labels = data["Target"].value_counts().index
sizes = data["Target"].value_counts().values
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
plt.title("Pie Chart of Class Distribution", color="#58508d")

plt.show()


# In[ ]:




