#!/usr/bin/env python3

from flask import Flask, request, redirect, url_for, flash, jsonify, abort
import numpy as np
import pickle as pickle
import json
import emoji
import sys

emojify = Flask(__name__)
modelfile = 'models/emojify_prediction.pickle'
model = pickle.load(open(modelfile, 'rb'))
model._make_predict_function()


emoji_dictionary = {"0": "\u2764\uFE0F",  
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

with open('word_to_index.json', 'r') as fp:
    word_to_index = json.load(fp)

def label_to_emoji(label):

    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)

# Converts words to incdieces to process
def sentences_to_indices(X, word_to_index, max_len):
    
    m = X.shape[0]                              
    
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):                             
        sentence_words =X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1
    
    return X_indices

@emojify.route('/emojify/api/v1.0', methods=['GET'])    
def makecalc():
    """
    Function run at each API call
    No need to re-load the model 
    """

    res = {} 

    if not request.args.get('text'):
        abort(400)

    sentence = request.args.get('text')
    if sentence:
        X = np.array([str(sentence)])
        max_len = 10
        X_test_indices = sentences_to_indices(X, word_to_index, max_len)
        res['text'] = (X[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))
        # returns a json file
        print(res)
    else:
        abort(403)
    return jsonify(res)

if __name__ == '__main__':
    # Model is loaded when the API is launched
    emojify.run(debug=True)
