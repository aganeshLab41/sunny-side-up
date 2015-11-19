import os,sys
file_path = os.path.dirname(os.path.abspath(__file__))
ssu_path = file_path.rsplit('/examples')[0]
sys.path.insert(0, ssu_path)

import json
import numpy as np
import pprint
import time

from datasets import word_vector_embedder as wv
from datasets import batch_data
from datasets import data_utils
from sys import argv

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, MaxPooling2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils,generic_utils
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python imdb_run.py
'''

def run_tests():

    #Get model name from command line
    script, model_name = argv 

    #Set num_epochs for training 
    num_epochs = 10

    #Set batch size for training
    batch_size = 128

    #Set boolean for if input needs to be transformed to 4D from 3D
    add_dim_to_input = False

    #Set input data and model based on command line input and model 
    if(model_name == 'char_cnn'):
        (imdbtr, imdbte),(tr_size, te_size) = char_input(batch_size=batch_size)
        model,sgd = char_cnn()
        add_dim_to_input = True

    elif(model_name == 'glove_cnn'):
        (imdbtr, imdbte),(tr_size, te_size) = word_embedding_input('glove',batch_size=batch_size)
        model,sgd = word_cnn('glove')
        add_dim_to_input = True

    elif(model_name == 'word2vec_cnn'):
        (imdbtr, imdbte),(tr_size, te_size) = word_embedding_input('word2vec',batch_size=batch_size)
        model,sgd = word_cnn('word2vec')
        add_dim_to_input = True

    elif(model_name == 'char_lstm'):
        (imdbtr, imdbte),(tr_size, te_size) = char_input(batch_size=batch_size)
        model = char_lstm()

    elif(model_name == 'glove_lstm'):
        (imdbtr, imdbte),(tr_size, te_size) = word_embedding_input('glove',batch_size=batch_size)
        model = word_lstm('glove')

    elif(model_name == 'word2vec_lstm'):    
        (imdbtr, imdbte),(tr_size, te_size) = word_embedding_input('word2vec',batch_size=batch_size)
        model = word_lstm('word2vec')

    else:       
        raise Exception('Input the correct model name')

    #Begin runs of training and testing    
    for e in range(num_epochs):
        print('-'*10)
        print('Epoch', e)
        print('-'*10)
        print("Training...")

        #Progress bar initialized for training data
        progbar = generic_utils.Progbar(tr_size)

        for X_batch, Y_batch in imdbtr():

            if(add_dim_to_input == True):
                X_batch = X_batch[:,np.newaxis]

            loss,acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("train loss", loss),("train acc",acc)])

        print("\n") 

        print("\nTesting...")
        
        #Progress bar initialized for testing data
        progbar = generic_utils.Progbar(te_size)

        for X_batch, Y_batch in imdbte():

            if(add_dim_to_input == True):
                X_batch = X_batch[:,np.newaxis]

            loss,acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("test loss", loss),("test acc",acc)])

        print("\n")


def word_embedding_input(embedding_type,batch_size=30,num_words=100):

    if(embedding_type == 'word2vec'):
        embedder = wv.WordVectorEmbedder('word2vec')
    else:
        embedder = wv.WordVectorEmbedder('glove')   

    (imdbtr, imdbte),(train_size, test_size) = batch_data.split_and_batch(
        data_loader=None,doclength=None,batch_size=batch_size,h5_path='imdb_split.hd5',
        transformer_fun=lambda x: embedder.embed_words_into_vectors(data_utils.tokenize(x),
            num_features=num_words),
        normalizer_fun=lambda x: data_utils.normalize(x,encoding=None,reverse=False),
        flatten=False)  

    return (imdbtr, imdbte),(train_size, test_size)

def char_input(batch_size=30):

    (imdbtr, imdbte),(train_size, test_size) = batch_data.split_and_batch(
        data_loader=None,doclength=None,batch_size=batch_size,h5_path='imdb_split.hd5')

    return (imdbtr, imdbte),(train_size, test_size)

def char_cnn():

    num_features = 67
    num_chars = 1014

    print('Build char cnn model...')
  
    #Set Parameters for final fully connected layers 
    fully_connected = [1024,1024,1]
    
    model = Sequential()

    #Input = #alphabet x 1014
    model.add(Convolution2D(256,num_features,7,input_shape=(1,num_features,num_chars)))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #Input = 336 x 256
    model.add(Convolution2D(256,1,7))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #Input = 110 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 108 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 106 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 104 X 256
    model.add(Convolution2D(256,1,3))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #34 x 256
    model.add(Flatten())

    #Fully Connected Layers

    #Input is 8704 Output is 1024 
    model.add(Dense(fully_connected[0]))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    #Input is 1024 Output is 1024
    model.add(Dense(fully_connected[1]))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    #Input is 1024 Output is 1
    model.add(Dense(fully_connected[2]))
    model.add(Activation('sigmoid'))
    
    #Stochastic gradient parameters as set by paper
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary")
    
    return model,sgd

def word_cnn(embedding_type):

    num_words = 99

    if(embedding_type=='word2vec'):
        num_features = 300
    else:   
        num_features = 200

    print('Build word cnn model...')
  
    #Set Parameters for final fully connected layers 
    fully_connected = [1024,1024,1]
    
    model = Sequential()

    #Input = #alphabet=(200 or 300) x 99
    model.add(Convolution2D(256,num_features,7,input_shape=(1,num_features,num_words)))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #Input = 31 x 256
    model.add(Convolution2D(256,1,7))
    
    #Input = 25 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 23 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 21 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 19 X 256
    model.add(Convolution2D(256,1,3))
    
    #17 x 256
    model.add(Flatten())

    #Fully Connected Layers

    #Input is 17*256=4352 Output is 1024 
    model.add(Dense(fully_connected[0]))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    #Input is 1024 Output is 1024
    model.add(Dense(fully_connected[1]))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    #Input is 1024 Output is 1
    model.add(Dense(fully_connected[2]))
    model.add(Activation('sigmoid'))
    
    #Stochastic gradient parameters as set by paper
    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary")
    
    return model,sgd

def word_lstm(embedding_type):
    
    num_words = 99

    if(embedding_type=='word2vec'):
        num_features = 300
    else:   
        num_features = 200

    print('Build word lstm model...')
    
    model = Sequential()

    model.add(LSTM(num_features,input_shape=(num_features,num_words),inner_init='orthogonal',forget_bias_init='one',activation='tanh',
        inner_activation='hard_sigmoid',truncate_gradient=-1))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    return model

def char_lstm():

    num_features = 67
    num_chars = 1014

    print('Build word lstm model...')
    
    model = Sequential()

    model.add(LSTM(num_features,input_shape=(num_features,num_chars),inner_init='orthogonal',forget_bias_init='one',activation='tanh',
        inner_activation='hard_sigmoid',truncate_gradient=-1))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    return model

if __name__=="__main__":
    run_tests()

