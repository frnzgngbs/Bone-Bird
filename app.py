import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

st.markdown(
    "<h1 style='text-align: center;'>Bird Categorical Group</h1>",
    unsafe_allow_html=True,
)

# Load the Keras model
model = load_model("bird-classifier.h5")

# Text labels for each measurement
st.markdown(
    "<h3 style='font-size: 18px;'>Enter Measurements (in mm):</h3>",
    unsafe_allow_html=True,
)
huml = st.number_input("Length of Humerus", min_value=0.0, key="huml")
humw = st.number_input("Diameter of Humerus", min_value=0.0, key="humw")
ulnal = st.number_input("Length of Ulna", min_value=0.0, key="ulnal")
ulnaw = st.number_input("Diameter of Ulna", min_value=0.0, key="ulnaw")
feml = st.number_input("Length of Femur", min_value=0.0, key="feml")
femw = st.number_input("Diameter of Femur", min_value=0.0, key="femw")
tibl = st.number_input("Length of Tibiotarsus", min_value=0.0, key="tibl")
tibw = st.number_input("Diameter of Tibiotarsus", min_value=0.0, key="tibw")
tarl = st.number_input("Length of Tarsometatarsus", min_value=0.0, key="tarl")
tarw = st.number_input("Diameter of Tarsometatarsus", min_value=0.0, key="tarw")
predict = st.button("Predict")

# Check if all fields are filled before predicting
if predict and not (
    huml
    and humw
    and ulnal
    and ulnaw
    and feml
    and femw
    and tibl
    and tibw
    and tarl
    and tarw
):
    st.error("Please fill in all measurements.")
elif predict:
    # Prepare input data
    measurements_array = np.array(
        [huml, humw, ulnal, ulnaw, feml, femw, tibl, tibw, tarl, tarw]
    )

    measurements_array = measurements_array.reshape(1, -1)

    st.write(measurements_array)

    measurements_array = np.expand_dims(measurements_array, axis=1)

    prediction = model.predict(measurements_array)

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
