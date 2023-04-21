
from flask import Flask, jsonify, make_response, request, redirect, render_template
from pathlib import Path
import sys

from transformers import AutoModelForSequenceClassification

from finbert.finbert import *
import finbert.utils as tools
import nltk
nltk.download('punkt')

project_dir = Path.cwd()#.parent
#pd.set_option('max_colwidth', -1)

best_model_folder = 'opinion_fact_v2_2'

cl_path = project_dir/'models'/best_model_folder
model = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)



app = Flask(__name__)
#vectorizer = pickle.load(open('../../vectorizer.sav', 'rb'))
#classifier = pickle.load(open('../../classifier.sav', 'rb'))
#classifier = pickle.load(open('../../classifier.sav', 'rb'))
#app.config['JSON_AS_ASCII'] = False

#@app.route('/sentiment', methods=['GET', 'POST'])
@app.route('/')
#def index():
#    return render_template('index.html')
def my_form():
    return render_template('index.html', variable='', headline='')

def encode_pred(pred):
    if pred == 'positive':
        return 'opinion'
    elif pred == 'negative':
        return 'fact'
    elif pred == 'neutral':
        return 'neutral'

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    if text:
        result = predict(text, model)
        result['prediction'] = result['prediction'].apply(encode_pred)
        result = result[['sentence', 'logit', 'prediction']]
        result = result.iloc[0, -1]
        print(result)
        return render_template('index.html', variable=str(result), headline=text)

if __name__ == '__main__':
    app.run()
#def sentiment_analysis():
    #if request.method == 'GET':
    #    text = request.args.get('text')
    #    if text:
    #        text_vector = vectorizer.transform([text])
    #        result = classifier.predict(text_vector)
    #        return make_response(jsonify({'sentiment': str(result[0]), 'text': text, 'status_code':200}), 200)
    #    return make_response(jsonify({'error':'sorry! unable to parse', 'status_code':500}), 500)
