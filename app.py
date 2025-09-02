# Streamlit Implementation
# Step 1: Import Libraries and Load the Model
import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('imdb_model.keras')

# Load the IMDB dataset and word index
max_features = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
  return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review

# Prediction Function
def predict_sentiment(review):
  preprocessed_input = preprocess_text(review)
  prediction = model.predict(preprocessed_input)
  sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
  return sentiment, prediction[0][0]

# Streamlit App
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')
user_input = st.text_input('Enter Review') # User Input

if st.button('Predict'):
  if user_input:
    sentiment, score = predict_sentiment(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {score}')
  else:
    st.warning('Please enter a review')



