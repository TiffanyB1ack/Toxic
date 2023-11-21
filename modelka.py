from joblib import dump, load
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re 
import matplotlib.pyplot as plt


df=pd.read_csv("labeled.csv")
df['comment']=df['comment'].str.lower()
df['comment'] = df['comment'].apply(lambda x:re.sub(r'[^\w\s]','', x)) 


vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(df['comment'])
vectorizer.vocabulary_
vectorizer.transform(df['comment']).toarray()
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df['comment'])

x_train,x_test,y_train,y_test = train_test_split(X,df['toxic'])

print('модель обучается')
model = SVC()
model.fit(x_train,y_train)
print('обучилась')

dump(model, 'filename.joblib') 