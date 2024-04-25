import streamlit as st
import pickle

pickle_in = open("bird-classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

st.title("Bird-Bone Classifier")
