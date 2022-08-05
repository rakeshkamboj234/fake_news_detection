import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app=Flask(__name__)

#Creating the model object
model = tf.keras.models.load_model("fake_news.h5")

# loading the pickle file
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Default route to be called
@app.route('/')
def home():
    # returns the home veiw
    return render_template('index.html')

# Route called when button will be clicked from the front-end
@app.route('/predict',methods=["POST"])
def predict():
    # Fetched values from the input controls
    int_features = [x for x in request.form.values()]
    print(int_features)
    #initial Max length
    max_length=0
    # Converting into array
    #final_features = np.array(int_features)
    
    # Convert the text into sequences
    sequence = tokenizer.texts_to_sequences(int_features)
    print(sequence)
    
    # Pad the sequences
    token_list = pad_sequences(sequence, maxlen=12468, truncating='post')
    # Get the probabilities of predicting a word
    prediction = model.predict(token_list)
    predicted = 'Real' if prediction[0][0] > .5 else 'Fake'
    return render_template('index.html', prediction_text = 'The Above news is  {}'.format(str(prediction)))

if __name__=='__main__':
    app.run()