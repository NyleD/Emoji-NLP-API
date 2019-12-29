import numpy as np
import emoji
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import pickle
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix


def read_glove_vecs(glove_file):
    with open (glove_file) as f: 
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def read_csv(filename = 'data/emojify_data.csv'):
    # need to store the file given somewhere
    phrase = []
    emoji = []

    with open (filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)
              
    
def print_predictions(X, pred):
    print()
    for i in range(X.shape[0]):
        print(X[i], label_to_emoji(int(pred[i])))
        

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

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    
    vocab_len = len(word_to_index) + 1             
    emb_dim = word_to_vec_map["cucumber"].shape[0]  
    
    emb_matrix = np.zeros((vocab_len, emb_dim))
   
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]


    embedding_layer = Embedding(vocab_len, emb_dim,trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):

    sentence_indices = Input(shape = input_shape, dtype = 'int32')

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    embeddings = embedding_layer(sentence_indices)

    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5,)(X)
    X = LSTM(128, return_sequences=False)(embeddings)
    X = Dropout(0.5,)(X)
    X = Dense(5)(X)
    X = Activation("softmax")(X)
    

    model = Model(inputs=sentence_indices,outputs=X)

    return model



def testAccuracy(X_test,word_to_index,maxLen,Y_test,model):
    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C = 5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test accuracy = ", acc)


def predict():  
    word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')
    X_train, Y_train = read_csv('train_emoji.csv')
    X_test, Y_test = read_csv('tesss.csv')
    maxLen = len(max(X_train, key=len).split())


    model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
    Y_train_oh = convert_to_one_hot(Y_train, C = 5)

    model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)
    

    X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
    Y_test_oh = convert_to_one_hot(Y_test, C = 5)
    loss, acc = model.evaluate(X_test_indices, Y_test_oh)
    print()
    print("Test accuracy = ", acc)
    
    
    
    pickle.dump(model,open('models/emojify_prediction.pickle', 'wb'))


predict()

