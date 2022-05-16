#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[46]:


data=pd.read_csv("C:/Users/rrukm/Downloads/data.csv",encoding= 'unicode_escape')
data


# In[47]:


data.info()


# # Process
# 
# 1. To Find Similarity between products
# 2. To Find similarity between Customers

# # To find Similarity between products:
# 1. Extract the Text Column (Description)
# 2. Drop The Duplicates
# 3. Do the Text Cleaning
#         a. Convert to lower case
#         b. Punctuation,short words Removal
#         c. POS tag
#         d. Lemmatization
# 4. Bag of Words (Convertion of text to numbers)
# 5. Apply Kmeans Algorithm (to find the unique products)
# 6. Find the Best Value of K

# In[48]:


data.Description.isna().sum()


# In[49]:


# Dropping Missing Values
data = data.dropna(subset = ['Description'])
data


# In[50]:


# Extraction of text data 
data_df=data[['Description']].copy()


# In[51]:


#Dropping Duplicates
data_df=data_df.drop_duplicates()


# In[52]:


#Lower Case coversion
data_df['Description_clean']=data.Description.str.lower()
data_df


# In[53]:


#Punctuation Removal
data_df['Description_clean']=data_df.Description_clean.str.replace("[^A-Z,a-z,0-9]"," ")
data_df


# In[54]:


#Removing words whose length is lesser than 2
data_df.Description_clean=data_df.Description_clean.apply(lambda x:' '.join([w for w in x.split() if len(w)>2]))


# In[55]:


#Removal of Stopwords
# POS tagging

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
stop_words=stopwords.words('english')


def remove_stopwords(rev):
    rev_tokenized=word_tokenize(rev)
    rev_new = ' '.join([i for i in rev_tokenized if i not in stop_words])
    return rev_new
data_df.Decription_clean=[remove_stopwords(r) for r in data_df['Description_clean']]
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')

lem=WordNetLemmatizer()
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# In[56]:


#Lemmatization
def lemm_sent(sentence):
    nltk_tagged=nltk.pos_tag(nltk.word_tokenize(sentence))
    wordnet_tag=map(lambda x:(x[0],nltk_tag_to_wordnet_tag(x[1])),nltk_tagged)
    lemmatized_sent=[]
    for word,tag in wordnet_tag:
        if tag is None:
            lemmatized_sent.append(word)
        else:
            lemmatized_sent.append(lem.lemmatize(word,tag))
    return ' '.join(lemmatized_sent)

data_df.Description_clean=data_df.Description_clean.apply(lambda x:lemm_sent(x))


# In[57]:


data_df.info()


# In[58]:


# Bag of Words (TFIDF)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_features = 5000)
ss=StandardScaler()
x=ss.fit_transform(x)
x=tfidf.fit_transform(data_df.Description_clean).toarray()

featureNames=tfidf.get_feature_names()
x


# In[59]:


#Applying K means algorithm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans


# In[60]:


Sum_of_squared_distances = []
list_k = list(range(1,18))
for i in list_k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(x)
    Sum_of_squared_distances.append(kmeans.inertia_)
plt.figure(figsize=(6,6))
plt.plot(list_k,Sum_of_squared_distances,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Sum of squared distances/Inertia') 
plt.title('Elbow Method For Optimal k')
plt.show()


# In[29]:


#K Value =15


# In[63]:


kkmeans=KMeans(n_clusters=15)
y=kkmeans.fit_predict(x)


# In[64]:


print(y)


# In[32]:


data1_new1=pd.get_dummies(y,prefix='product')


# In[256]:


merged = pd.merge(left=data_df, left_index=True,right=data1_new1, right_index=True,
                  how='right')


# In[257]:


merged


# In[258]:


data_mer=pd.merge(left=data, left_index=True,right=merged, right_index=True,
                  how='right')


# In[259]:


data_mer.Country.unique()


# In[260]:


data_mer.fillna(data_mer['Country'].mode()[0],inplace=True)


# In[233]:


data_mer=pd.get_dummies(data_mer,columns=['Country'])


# In[271]:


data_mer


# # Segmention done by transaction 
# 
# ***Using groupby***

# In[281]:


merger_data=data_mer.groupby(['CustomerID','Country'],as_index = False).sum()


# In[ ]:


df2 = df.groupby(['Courses','Duration'],as_index = False).sum().pivot('Courses','Duration').fillna(0)
print(df2)


# In[242]:


merger_data=pd.DataFrame(merger_data.iloc[:152])


# In[360]:


merger_data=pd.get_dummies(merger_data,columns=['Country'])


# In[362]:


merger_data


# In[365]:


X=merger_data.iloc[:152].values
X


# In[366]:


wcss = []
for i in range(1, 14):
    kmeans10 = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans10.fit(X)
   #appending the WCSS to the list (kmeans.inertia_ returns the WCSS value for an initialized cluster)
    wcss.append(kmeans10.inertia_)  
#Plotting The Elbow graph
plt.plot(range(1, 14), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[367]:


kkmeans1=KMeans(n_clusters=3)
m=kkmeans1.fit_predict(X)


# In[368]:


m


# In[381]:


# Visualising the clusters
figure = plt.figure(figsize=[10, 7])
#Scatter plotting for (x,y) with label 1 as Cluster 1 in color c = red and points in size s = 50
plt.scatter(X[m == 0, 0], X[m == 0, 1], s = 20, c = 'red', label = 'Cluster 1')
#Scatter plotting for (x,y) with label 2 as Cluster 2 in color c = blue and points in size s = 50
plt.scatter(X[m == 1, 0], X[m == 1, 1], s = 20, c = 'blue', label = 'Cluster 2')
plt.scatter(X[m == 2, 0], X[m == 2, 1], s = 20, c = 'green', label = 'Cluster 3')

#Scatter plotting the centroids with label = 'Centroids' in color c = cyan and points in size s = 100
plt.scatter(kkmeans1.cluster_centers_[:, 0], kkmeans1.cluster_centers_[:, 1], s = 15, c = 'cyan', label = 'Centroids')
plt.xticks(ticks=np.arange(0,20000,2000), size=12)

# Changing y-ticks value using an array of 19 value with step size of 2
plt.yticks(ticks=np.arange(0, 150,10), size=12)

plt.title('Different Products')
plt.xlabel('Customers')
plt.ylabel('Products')
plt.legend()
plt.show()


# # Conclusion
# 
# ***There are 3 Types of customers that purchase similar type of products***

# In[ ]:




