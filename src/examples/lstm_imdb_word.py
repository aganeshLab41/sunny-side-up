from __future__ import absolute_import
from __future__ import print_function

import os,sys
file_path = os.path.dirname(os.path.abspath(__file__))
ssu_path = file_path.rsplit('/examples')[0]
sys.path.insert(0, ssu_path)

import json
import numpy as np
import pprint
import time

from datasets import amazon_reviews
from datasets import batch_data
from datasets import data_utils
from datasets import word_vector_embedder as wv

#from word_vector_embedder import WordVectorEmbedder
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
Model below is based on paper by Xiang Zhang "Character-Level Convolutional
Networks for Text Classification" (http://arxiv.org/abs/1509.01626) paper was
formerly known as "Text Understanding from Scratch" (http://arxiv.org/pdf/1502.01710v4.pdf) 

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python lstm_imdb_word.py
'''

def model_defn(num_words,model_dim):

    print('Build model...')
    
    model = Sequential()

    #model.add(Convolution2D(256,67,7,input_shape=(1,67,1014)))
    #model.add(Convolution1D(256,7,input_shape=(67,1014)))

    model.add(LSTM(300,input_shape=(num_words,model_dim),init='orthogonal',forget_bias_init='one',activation='tanh',
        inner_activation='hard_sigmoid',truncate_gradient=-1))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    return model

if __name__=="__main__":

    #Set batch size for input data
    batch_size = 30

    #Set the number of epochs to run
    num_epochs = 10

    num_words = 100

    embedder = wv.WordVectorEmbedder('word2vec')

    #Import model 
    model = model_defn(num_words,embedder.num_features())

    (imdbtr, imdbte),(train_size, test_size) = batch_data.split_and_batch(
        data_loader=None,doclength=None,batch_size=batch_size,h5_path='imdb_split.hd5',
        transformer_fun=lambda x: embedder.embed_words_into_vectors(data_utils.tokenize(x),
            num_features=num_words),
        normalizer_fun=lambda x: data_utils.normalize(x,encoding=None,reverse=False),
        flatten=False)
    
    print("Train size", train_size)
    print("Test size", test_size)

    text,result = imdbtr().next()
    print("text shape", text.shape)
    print("result shape", result.shape)

    

    #Begin runs of training and testing    
    for e in range(num_epochs):
        print('-'*10)
        print('Epoch', e)
        print('-'*10)
        print("Training...")

        #Progress bar initialized for training data
        progbar = generic_utils.Progbar(train_size)

        for X_batch, Y_batch in imdbtr():

            #X_batch = X_batch[:,np.newaxis]
            loss,acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("train loss", loss),("train acc",acc)])

        print("\n") 

        print("\nTesting...")
        
        #Progress bar initialized for testing data
        progbar = generic_utils.Progbar(test_size)

        for X_batch, Y_batch in imdbte():

            #X_batch = X_batch[:,np.newaxis]
            loss,acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("test loss", loss),("test acc",acc)])

        print("\n")
    



