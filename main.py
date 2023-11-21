from flask import Flask, render_template, request, render_template, jsonify, redirect
import numpy as np
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from keras.models import model_from_json

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from joblib import load
import re
from pymystem3 import Mystem
import pickle
import json
from keras.preprocessing.text import Tokenizer

# Load the saved tokenizer object from a file
with open('C:/Users/User/Desktop/мага/toxicc/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the saved word index from a file
with open('C:/Users/User/Desktop/мага/toxicc/word_index.json', 'r') as handle:
    word_index = json.load(handle)
    
def text_cleaner(text):
    tokenized_text = word_tokenize(text, language='russian')
    clean_text = [word.lower() for word in tokenized_text if word not in punctuation and word != '\n']
    r = re.compile("[а-яА-Я]+")
    russian_text = ' '.join([w for w in filter(r.match, clean_text)])
    return russian_text


lemmatizator = Mystem()
dict_size=28064

max_comment_length = 250

model = load('C:/Users/User/Desktop/мага/toxicc/modelka.joblib')

app = Flask(__name__)

# перенаправление на страницу ввода
@app.route("/")
def main_page():
    return redirect("/input")

@app.route("/input", methods=["GET"])
def input_page():
    return render_template("index.html")

@app.route("/input", methods=["POST"])
def get_input():
    
    
    phrase = request.get_json()["phrase"]
    #example = 'прекрасный пример полного отсутствия мозга'
    clean_example = text_cleaner(phrase)
    lemm_example = ' '.join(lemmatizator.lemmatize(clean_example))
    array_example = np.array([lemm_example])
    seq_example = tokenizer.texts_to_sequences(array_example)
    max_comment_length = 250
    pad_example = pad_sequences(seq_example, maxlen=max_comment_length)
    pred_example = model.predict(pad_example)
    if  pred_example[0][0]>=0.5:  
        res='toxic'
        pr=round(pred_example[0][0]*100,4)
    else:
        res='not toxic'
        pr=100-round(pred_example[0][0]*100,4)
    s=res+' с точностью '+ str(pr)+'%'
    return jsonify({"message" : s}) # тут фразу заменить на результат

# запуск приложения
if __name__ == "__main__":
    app.run()
