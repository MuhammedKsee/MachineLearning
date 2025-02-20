import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk import word_tokenize
from nltk.corpus import stopwords
import re

dataset = pd.read_csv('LogisticRegression/dataset.csv')

dataset.sort_values("Label",inplace=True)
dataset = dataset.drop(columns="B")

dataset.drop_duplicates(subset="Body",keep=False,inplace=True)

def optimization(dataset):
    dataset = dataset.dropna()
    stop_words = set(stopwords.words('turkish'))
    punctuation = ['.',',','?','!','"','(',')','[',']','{','}','-',
                   '_','+','=','*','/','\\','|','@','#','$','%','^',
                   '&','*','~','`',';',':','<','>','...','..','....',"0","1","2","3","4","5","6","7","8","9","\"","'"]
    stop_words.update(punctuation)

    for ind in dataset.index:
        body = dataset['Body'][ind]
        body = body.lower()
        body = re.sub(r'http\S+', '', body)
        body = (" ").join([word for word in body.split() if word not in stop_words])
        body = "".join([char for char in body if char not in punctuation])
        dataset['Body'][ind] = body
        return dataset

dataset = optimization(dataset)

comment_machine = dataset[dataset["Label"] == 0]
comment_human = dataset[dataset["Label"] == 1]

tfIdf = TfidfVectorizer(binary=False,ngram_range=(1,3))

comment_machine_vec = tfIdf.fit_transform(comment_machine["Body"])
comment_human_vec = tfIdf.fit_transform(comment_human["Body"].tolist())

print(comment_machine_vec)    

x = dataset.loc[:,"Body"]
y = dataset.loc[:,"Label"]
x_vec = tfIdf.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train_vec , x_test_vec , y_train , y_test= train_test_split(x_vec,y,test_size=0.2,random_state=0)

logisticRegression = LogisticRegression()

logisticRegression.fit(x_train_vec,y_train)

pickle.dump(logisticRegression,open("LogisticRegression/model","wb"))
print("Model is trained and saved")

pickle.dump(tfIdf,open("LogisticRegression/vectorizer","wb"))    
print("Vectorizer is saved")


