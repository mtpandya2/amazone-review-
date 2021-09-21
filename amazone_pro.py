# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:45:19 2021

@author: Nirzari Pandya
"""

import pandas as pd

import numpy as np

df=pd.read_csv("c:/Users/Nirzari Pandya/Desktop/Nizu iphone/amazone project/Reviews.csv")

df.head()

df.columns

df['helpful%']=np.where(df['HelpfulnessDenominator']>0,df['HelpfulnessNumerator']/
         df['HelpfulnessDenominator'],-1)

df.head()


df['helpful%'].unique()


df['%upvote']=pd.cut(df['helpful%'],bins=[-1,0,0.2,0.4,0.6,0.8,1],labels=['Empty',
                                                           '0-20%','20-40%'
                                                          ,'40-60%',
                                                          '60-80%','80-100%'])

df.head()

df.groupby(['Score','%upvote']).agg('count')


df_s=df.groupby(['Score','%upvote']).agg({'Id':'count'}).reset_index()

pivot=df_s.pivot(index='%upvote',columns='Score')

import seaborn as sns

sns.heatmap(pivot,annot=True,cmap='YlGnBu')

df['Score'].unique()

df2=df[df['Score']!=3]

x=df2['Text']

df2['Score'].unique()


y_dict={1:0,2:0,4:1,5:1}

y=df2['Score'].map(y_dict)


from sklearn.feature_extraction.text import CountVectorizer

c=CountVectorizer(stop_words='english')

x_c=c.fit_transform(x)

x_c.shape[1]


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_c,y)

x_train.shape


from sklearn.linear_model import LogisticRegression

loc=LogisticRegression()

ml=loc.fit(x_train,y_train)

ml.score(x_test,y_test)

w=c.get_feature_names()
w

coef=ml.coef_.tolist()[0]
coef

coef_df=pd.DataFrame({'word':w,'coof':coef})
coef_df

coef_df.sort_values(['coof','word'],ascending=False)


def text_fit(x,y,nlp_model,ml_model,coef=1):
    
    x_c=nlp_model.fit_transform(x)
    
    print('feaatures:{}'.format(x_c.shape[1]))
    
    x_train,x_test,y_train,y_test=train_test_split(x_c,y)
    ml=ml_model.fit(x_train,y_train)
    
    acc=ml.score(x_test,y_test)
    
    print(acc)
    
    if(coef==1):
        
    
        w=c.get_feature_names()
        coef=ml.coef_.tolist()[0]

    
        coef_df=pd.DataFrame({'word':w,'coof':coef})  
    
    
        coef_df.sort_values(['coof','word'],ascending=False)
    
        print('Top 20 positive words')
    
        print(coef_df.head(20))
    
        print('Top 20 negative words')
    
        print(coef_df.tail(20))
    
from sklearn.feature_extraction.text import CountVectorizer

c=CountVectorizer(stop_words='english')

from sklearn.linear_model import LogisticRegression

text_fit(x,y,c,LogisticRegression())


from sklearn.metrics import confusion_matrix,accuracy_score

def fit(x,y,nlp_model,ml_model):
    
    x_c=nlp_model.fit_transform(x)
    
    x_train,x_test,y_train,y_test=train_test_split(x_c,y)
    ml=ml_model.fit(x_train,y_train)
    
    predictions=ml.predict(x_test)
    
    cm=confusion_matrix(predictions,y_test)
    
    print(cm)
    
    acc=accuracy_score(predictions, y_test)
    
    print(acc)



c=CountVectorizer()

lr=LogisticRegression()


fit(x,y,c,lr)

from sklearn.dummy import DummyClassifier

c=CountVectorizer()

text_fit(x, y, c, DummyClassifier(),0)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')


text_fit(x, y, tfidf,lr,0)


fit(x, y, c,lr)


df.head()


data=df[df['Score']==5]

data.head()

data['%upvote'].unique()    


data2=data[data['%upvote'].isin(['80-100%', '60-80%','20-40%', '0-20%'])]


data2.head()


x=data2['Text']

data2['%upvote'].unique()    

y_dict={'80-100%':1, '60-80%':1,'20-40%':0, '0-20%':0}

y=data['%upvote'].map(y_dict)

y.value_counts()



from sklearn.feature_extraction.text import TfidfVectorizer

tf=TfidfVectorizer()

x_c=tf.fit_transform(x)

!pip install Tensorflow

from imblearn.over_sampling import RandomOverSampler()


os=RandomOverSampler()

x_train_res,y_train_res=os.fit_sample(x_c,y)

from collections import Counter

print("original dataset shape {}".format(Counter(y)))


print("resampled dataset shape {}".format(Counter(y_train_res)))


from sklearn.linear_model import LogisticRegression

log_class=LogisticRegression()

np.arange(-2,3)

grid={'c':10.0**np.arange(-2,3),'penalty':['l1','l2']}



from sklearn.model_selection import GridSearchCV

clf=GridSearchCV(estimator=log_class,param_grid=grid,cv=5,n_jobs=-1,scoring='f1_macro')
clf.fit(x_train_res,y_train_res)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_c,y)

pred=clf.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score

confusion_matrix(y_test,pred)


accuracy_score(y_test,pred)


