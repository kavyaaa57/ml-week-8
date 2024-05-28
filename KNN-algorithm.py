import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset["data"], dataset["target"], random_state=0)

# Sidebar for user input
st.sidebar.header('User Input Parameters')

def user_input_features():
    n_neighbors = st.sidebar.slider('Number of neighbors (k)', 1, 10, 1)
    return n_neighbors

n_neighbors = user_input_features()

# Train the KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=n_neighbors)
kn.fit(X_train, y_train)

# Display predictions for each test sample
st.title('Iris Dataset KNN Classification')
st.write('This app performs classification on the Iris dataset using the K-Nearest Neighbors (KNN) algorithm.')

st.subheader('Test Sample Predictions')
for i in range(len(X_test)):
    x = X_test[i]
    x_new = np.array([x])
    prediction = kn.predict(x_new)
    st.write(f"Test Sample {i+1}:")
    st.write(f"TARGET = {y_test[i]} ({dataset['target_names'][y_test[i]]}), PREDICTED = {prediction[0]} ({dataset['target_names'][prediction[0]]})")

# Display the accuracy of the model
accuracy = kn.score(X_test, y_test)
st.subheader('Model Accuracy')
st.write(f"The accuracy of the model is: {accuracy * 100:.2f}%")
