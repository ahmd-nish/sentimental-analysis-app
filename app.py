from flask import Flask, request
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
import pickle


app = Flask(__name__)

# Load the tokenizer and the model
tokenizer = pickle.load(open("./tokenizer.pkl", "rb"))
model = tf.keras.models.load_model("./model.h5")

# Load the index_to_classes mapping
index_to_classes = pickle.load(open("./index_to_class.pkl", "rb"))

def predict_sentiment(text):
    # Preprocess the input text
    token = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences([token[0]], maxlen=50, padding="post")
    
    # Predict the sentiment
    p = model.predict(padded)
    sentiment = index_to_classes[np.argmax(p).astype('int')]
    
    # Return the sentiment
    return sentiment


@app.route("/predict", methods=["POST"])
def predict():
    # Get the text to be analyzed from the request
    text = request.form.get("text")
    
    # Predict the sentiment of the text
    sentiment = predict_sentiment(text)
    
    # Return the sentiment as a JSON response
    return {"sentiment": sentiment}

if __name__ == "__main__":
    app.run()