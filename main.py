import streamlit as st
import numpy as np
import pandas as pd
import pickle as pkl
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing import sequence
import bs4
from sklearn.feature_extraction.text import TfidfVectorizer



lg_model=pkl.load(open('C:/Users/sarat/DF 2009/portfolio/lg_review_1.pickle','rb'))

vectorizer=pkl.load(open('C:/Users/sarat/DF 2009/portfolio/tfidf_rev.pickle','rb'))

# vectorizer_2=TfidfVectorizer()



def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

def process_data(sentance):
  sentance = re.sub(r"http\S+", "", sentance)
  # sentance = bs4.BeautifulSoup(sentance, 'lxml').get_text()
  sentance = decontracted(sentance)
  # removing extra spaces and numbers
  sentance = re.sub("\S*\d\S*", "", sentance).strip()
  # removing non alphabels
  sentance = re.sub('[^A-Za-z]+', ' ', sentance)
  text = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
  return text



st.title('Food Review Analysis')

page_bg_img = '''
<style>
body {
background-image: url('https://images.unsplash.com/photo-1614208533225-e96778447b80?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1yZWxhdGVkfDE3fHx8ZW58MHx8fHw%3D&auto=format&fit=crop&w=500&q=60');
background-repeat: no-repeat;
background-size: cover;
}
.block-container {
    backdrop-filter: blur(10px);
}
.markdown-text-container > h1{text-align:center}
.css-1sls4c0 {
    background:none;
        backdrop-filter: blur(10px);
}
.stApp {
    background:transparent;
     background:none;
        backdrop-filter: blur(10px);
}
.css-hi6a2p {
    padding-left: 1rem;
    padding-right: 1rem;
.st-cm {
    background-color: rgb(240 242 246);
}
}
</style>
'''


st.markdown(page_bg_img, unsafe_allow_html=True)


box1=st.text_area("Enter Review")
# print(box1)

def class_pred(pred):
    if pred[0][1] >= 0.60 and pred[0][0] <= 0.40:
        label='positive'

    elif pred[0][0] >= 0.60 and pred[0][1] <= 0.40:
        label = 'negative'

    else:
        label = 'neutral'
    return label

def prob(pred,label):
    if label=='positive':
        a=round(pred[0][1]*100, 2)
        r="The following review is " + str(label) + " with " + str(a) + "% of positivity"
        return r
    elif label=='negative':
        a=round(pred[0][0]*100, 2)
        r="The following review is " + str(label) + " with " + str(a) + "% of negativity"
        return r
    else:
        a=round(pred[0][1]*100, 2)
        b=round(pred[0][0]*100, 2)
        r="The following review is neutral with " + str(a) + "% of positivity and " + str(b) + "% of negativity"
        return r

if st.button('Predict'):
    wrd_text = process_data(box1)
    # print("1"+wrd_text)
    lis=[]
    lis.append(wrd_text)
    vect_form = vectorizer.transform(lis).toarray()

    # print(vect_form)
    pred = lg_model.predict_proba(vect_form)
    # print(pred)
    label=class_pred(pred)
    res=prob(pred,label)

    st.success(box1)
    st.success(res)
