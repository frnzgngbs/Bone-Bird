import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title("Bird-Bone Measurement Input")

pickle_in = open("bird-classifier.pkl", "rb")
model = pickle.load(pickle_in)

# Text labels for each measurement
# Text labels for each measurement
st.subheader("Enter Measurements (in mm):")
huml = st.number_input("Length of Humerus", min_value=0.0)
humw = st.number_input("Diameter of Humerus", min_value=0.0)
ulnal = st.number_input("Length of Ulna", min_value=0.0)
ulnaw = st.number_input("Diameter of Ulna", min_value=0.0)
feml = st.number_input("Length of Femur", min_value=0.0)
femw = st.number_input("Diameter of Femur", min_value=0.0)
tibl = st.number_input("Length of Tibiotarsus", min_value=0.0)
tibw = st.number_input("Diameter of Tibiotarsus", min_value=0.0)
tarl = st.number_input("Length of Tarsometatarsus", min_value=0.0)
tarw = st.number_input("Diameter of Tarsometatarsus", min_value=0.0)
predict = st.button("Predict")

if predict:
    # Prepare input data
    measurements_array = np.array(
        [huml, humw, ulnal, ulnaw, feml, femw, tibl, tibw, tarl, tarw]
    )

    column_names = [
        "huml",
        "humw",
        "ulnal",
        "ulnaw",
        "feml",
        "femw",
        "tibl",
        "tibw",
        "tarl",
        "tarw",
    ]

    measurements_array = measurements_array.reshape(1, -1)

    st.write(measurements_array)

    measurements_array = np.expand_dims(measurements_array, axis=1)

    prediction = model.predict(measurements_array)

    st.subheader("Mao ni ang percentage sa kung unsa na type")
    st.write(prediction)

    predicted_class_index = np.argmax(prediction)

    if predicted_class_index == 3:
        st.subheader("Swimming Birds")
    elif predicted_class_index == 5:
        st.subheader("Wading Birds")
    elif predicted_class_index == 4:
        st.subheader("Terrestrial Birds")
    elif predicted_class_index == 1:
        st.subheader("Raptors")
    elif predicted_class_index == 0:
        st.subheader("Scansorial Birds")
    elif predicted_class_index == 2:
        st.subheader("Singing Birds")
