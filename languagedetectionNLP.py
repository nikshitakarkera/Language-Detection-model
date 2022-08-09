import pandas as pd
import numpy as np
import re 
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")
data=pd.read_csv("../Language Detection.csv", encoding='utf-8')# reading data from csv files
#data.head(10) run in console to check if the csv is loaded or not
data["Language"].value_counts()#count of the test set
X=data["Text"] # all the text saved in x
y=data["Language"] #all the 

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
data_list=[]
for text in X:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(data_list).toarray()
X.shape # (10337,39419)
#print(x)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)    
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy is :",ac)

plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()

def predict(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     print("The langauge is in",lang[0]) # printing the language

predict("Cómo va el día")

