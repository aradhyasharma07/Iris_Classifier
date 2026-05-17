import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

flower_names = [
    "Iris Setosa",
    "Iris Versicolor",
    "Iris Virginica"
]

st.title("🌸 Iris Flower Predictor")

st.write("Machine Learning Classification using KNN")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict Flower"):

    features = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(features)

    result = flower_names[prediction[0]]

    st.success(f"Predicted Flower: {result}")