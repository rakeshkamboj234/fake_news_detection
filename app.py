import pickle
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import regex


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
    features = [regex.sub(r'[^a-zA-Z\s]','',x).lower() for x in request.form.values()]
    # convert to lowercase
    text = str(features[0]).lower()  
    # remove single characters
    text = regex.sub(pattern=r'\s+[a-zA-Z]\s+',repl='',string = text) 
    # Remove URls, whitespace characters  
    text = regex.sub(r'https?://\S+|www\.\S+',repl='',string = text)  
    # Removes all the special characters, digits from 0-9 and Capital Letters  
    text = regex.sub(r'[^a-z\s]',' ',string = text)
    # Substituting multiple spaces with single space
    text = regex.sub(r'\s+', ' ', string = text)  

    print([text])
    # Convert the text into sequences
    sequence = tokenizer.texts_to_sequences([text])
    print("sequence", sequence)
    # Pad the sequences
    token_list = pad_sequences(sequence, maxlen=12468, truncating='post')
    print(token_list)
    # Get the probabilities of predicting a word
    prediction = model.predict(token_list)
    predicted = 'Real' if prediction[0][0] > .5 else 'Fake'
    return render_template('index.html', prediction_text = 'The Above news is  {}'.format(str(predicted)))



if __name__=='__main__':
    app.run()