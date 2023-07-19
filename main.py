from sentence_transformers import SentenceTransformer


import streamlit as st
import pickle


# load models

classifier_model = pickle.load(open("svc_spam_classifier.pickle","rb"))
sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")

# spam classification function


def is_spam(message):
    message_encoded = sentence_encoder.encode([message])
    prediction = classifier_model.predict(message_encoded)[0]
    return True if prediction == 1 else False


# UI
st.title("SMS Spam Detector 🕵🏼")

sms = st.text_area(label="",placeholder="Your SMS text goes here..")
clicked = st.button("Find Out!")

# handle button click
if clicked:
    if len(sms) == 0:
        st.write("Please put in some text")
    else:
        # check spam
        if is_spam(sms):
            st.write("Spam.")
        else:
            st.write("Not Spam")