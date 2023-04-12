import streamlit as st
import joblib
import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# load the model from disk
loaded_model = joblib.load('james_model.sav')

st.title("Welcome to Your App !")

uploaded_file = st.file_uploader("Kindly , Choose a file")
if uploaded_file is not None:
  df = pd.read_excel(uploaded_file, dtype=str)
  df.dropna(inplace=True)

  # Preprocess the text data
  vectorizer = CountVectorizer(stop_words='english', lowercase=False)
  X = vectorizer.fit_transform(df['Asset Description'])
  y = df['SFG20 Task Code'].values

  # load the model from disk
  loaded_model = joblib.load('james_model.sav.zip')

  # Make predictions on the testing set
  predictions = loaded_model.predict(X)

  st.write('Predictions:', predictions)
