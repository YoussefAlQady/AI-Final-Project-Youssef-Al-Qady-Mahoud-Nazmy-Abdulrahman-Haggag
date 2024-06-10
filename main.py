import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

data = pd.read_csv("train.csv")

x = data["sms"] 
y = data['label'] 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

vectorizer = CountVectorizer().fit(x_train)
sms_train_vectorized = vectorizer.transform(x_train)
sms_test_vectorized = vectorizer.transform(x_test)

clfr = MultinomialNB()
clfr.fit(sms_train_vectorized, y_train)

prediction = clfr.predict(sms_test_vectorized)
acc = metrics.accuracy_score(y_test, prediction)

st.title("Spam or Ham Detection")
st.image("spamham.jpg", use_column_width=True)
st.text("This datascience model uses Naive Bayes to predict sms messages")

input_text = st.text_input("Input text below", "")
predict = st.button("Find Result")

if st.button("Classify"):
    input_vectorized = vectorizer.transform([input_text])
    prediction = clfr.predict(input_vectorized)[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    if prediction == 1:
        st.error(f"'{input_text}' is classified as {result}")
    else:
        st.success(f"'{input_text}' is classified as {result}")

