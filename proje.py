# -*- coding: utf-8 -*-
"""
Created on Sun May 16 17:26:48 2021

@author: Caner
"""

import tweepy

from textblob import TextBlob
from sklearn.metrics import f1_score
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re 
import nltk
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
plt.style.use('fivethirtyeight') 

nltk.download('stopwords')#103..video türkçe stop word için başka bir şey yapman gerekiyor
from nltk.corpus import stopwords

result=pd.read_csv("C:/Users/P2/Desktop/pyton/twitter/result.csv")
derlem=[] 


def cleanTxt(text):
    text=re.sub(r'https?:\/\/\S+','',str(text))#remove the hyperlink
    #text=re.sub("^[a-zA-Z0-9ğüşöçİĞÜŞÖÇ]+$",' ',text)
    text=re.sub('[\W]',' ',text)
    text=re.sub(r'@[A-Z-z0-9]+','',str(text))#removed @mentions
    text=re.sub(r'#','',str(text))#removin hashtag symbol
    text=str(text).lower()
    text = str(text).split() 
    
    text = [ps.stem(kelime) for kelime in text if not kelime in set(stopwords.words('turkish'))]
    text = ' '.join(text)
    derlem.append(text)
    return text

#cleaning the text

result['Tweets']=result['Tweets'].apply(cleanTxt)




#--------------------------------------------------------------
#FeatureExtraction(Bag of Words)
#--------------------------------------------------------------

from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer(max_features=2000)
X = cv.fit_transform(derlem).toarray()#bağımsız değişken 
y = result.iloc[:,2].values #bağımlı değğişken 

#---------------------------------------------------------------
#Train-Test Bölünmesi
#---------------------------------------------------------------

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)

#---------------------------------------------------------------
#Naive Bayes
#-----------------------------------------------------------------------

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train, y_train)

y_pred=gnb.predict(x_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print('GNB') 
print(cm)

#----------------------------------------------------------------
#SVM(kernel=linear)
#----------------------------------------------------------------

from sklearn.svm import SVC
svc = SVC(kernel='linear')
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('SVM(kernel=linear)') 
print(cm)



#----------------------------------------------------------------
#SVM(kernel=rbf)
#----------------------------------------------------------------

from sklearn.svm import SVC
svc = SVC(kernel='rbf')
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('SVM(kernel=rbf)') 
print(cm)

#----------------------------------------------------------------

#----------------------------------------------------------------
#SVM(kernel=rbf)
#----------------------------------------------------------------

from sklearn.svm import SVC
svc = SVC(kernel='sigmoid')
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('SVM(kernel=sigmoid)') 
print(cm)

#----------------------------------------------------------------
#Random Forest(entropy)
#----------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('Random Forest(entropy)') 
print(cm)

f1=f1_score(y_test, y_pred)
print('Random Forest(entropy) f1:')
print(f1)


#----------------------------------------------------------------
#Random Forest(gini)
#----------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='gini')
rfc.fit(x_train, y_train)

y_pred = svc.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('Random Forest(gini)') 
print(cm)

f1=f1_score(y_test, y_pred)
print('Random Forest(gini) f1:')
print(f1)



#--------------------------------------
#K-NN
#-------------------------------------

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10,metric='minkowski')

knn.fit(x_train, y_train)

y_pred=knn.predict(x_test)

cm = confusion_matrix(y_test,y_pred)
print('K-NN') 
print(cm)


f1=f1_score(y_test, y_pred)
print('K-NN f1:')
print(f1)

from sklearn.metrics import f1_score

















