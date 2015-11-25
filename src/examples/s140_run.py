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
from datasets.sentiment140 import Sentiment140
from sys import argv
from datasets.batch_data import BatchIterator, DataIterator

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
from sklearn.metrics import confusion_matrix

'''
THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python s140_run.py
'''

def run_tests():

    #Get model name from command line
    script, model_name = argv 

    #Set num_epochs for training 
    num_epochs = 10

    #Set batch size for training
    batch_size = 128

    #Set predefined window of words to look at for word related models/embeddings
    num_words = 50

    #Set boolean for if input needs to be transformed to 4D from 3D
    add_dim_to_input = False

    #Set boolean for if sgd learning rate needs to be changed every few epochs
    halve_lr = False

    #Set input data and model based on command line input and model 
    if(model_name == 'char_cnn'):
        (s140tr, s140te),(tr_size, te_size) = char_input(batch_size=batch_size)
        model,sgd = char_cnn()
        add_dim_to_input = True
        halve_lr = True

    elif(model_name == 'glove_cnn'):
        (s140tr, s140te),(tr_size, te_size) = word_embedding_input('glove',batch_size=batch_size,num_words=num_words)
        model,sgd = word_cnn('glove',num_words)
        add_dim_to_input = True
        halve_lr = True

    elif(model_name == 'word2vec_cnn'):
        (s140tr, s140te),(tr_size, te_size) = word_embedding_input('word2vec',batch_size=batch_size,num_words=num_words)
        model,sgd = word_cnn('word2vec',num_words)
        add_dim_to_input = True
        halve_lr = True

    elif(model_name == 'char_lstm'):
        (s140tr, s140te),(tr_size, te_size) = char_input(batch_size=batch_size,chars_reversed=False)
        model = char_lstm()

    elif(model_name == 'glove_lstm'):
        (s140tr, s140te),(tr_size, te_size) = word_embedding_input('glove',batch_size=batch_size,num_words=num_words)
        model = word_lstm('glove',num_words)

    elif(model_name == 'word2vec_lstm'):    
        (s140tr, s140te),(tr_size, te_size) = word_embedding_input('word2vec',batch_size=batch_size,num_words=num_words)
        model = word_lstm('word2vec',num_words)

    else:       
        raise Exception('Input the correct model name')

    #Statistics Initialization 
    train_times = []
    test_accuracies = []
    test_confusions = []
    costs = []    
    save_path = 's140/'+ model_name + '/model.json'
    model_weights = 's140/'+ model_name + '/model_weights.hd5'

    #Begin runs of training and testing    
    for e in range(num_epochs):
        print('-'*10)
        print('Epoch', e)
        print('-'*10)
        print("Training...")

        #Initialize variables for collecting minibatch cost stats and starting train timer
        costs.append([])
        train_times.append({})
        train_times[e]['start'+'_epoch_'+str(e)] = time.time()

        if(halve_lr):
            if(e % 3 == 2):
                sgd.lr.set_value(sgd.lr.get_value()/2)

        #Progress bar initialized for training data
        progbar = generic_utils.Progbar(tr_size)

        #Initialize train stats variables
        trainCounter = 0

        for X_batch, Y_batch in s140tr:

            if(add_dim_to_input == True):
                X_batch = X_batch[:,np.newaxis]

            #print("Debugging X_batch size", X_batch.shape)    
            #print("Debugging Y_batch size", Y_batch.shape)

            loss,acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("train loss", loss),("train acc",acc)])

            trainCounter = trainCounter + 1

            #Record costs after each minibatch
            costs[e].append(float(loss))

        print("\n") 

        #Record how long training took per epoch
        train_times[e]['end'+'_epoch_'+str(e)] = time.time()
        
        #Save the model every epoch into hd5 file    
        model.save_weights(model_weights,overwrite=True)

        print("\nTesting...")

        #Initialize variables for collecting test related stats
        testCounter = 0
        testAccVal = 0

        #Confusion matrix initializations
        y_true = []
        y_test = []
        conf_matrix = np.zeros((2,2))
        
        #Progress bar initialized for testing data
        progbar = generic_utils.Progbar(te_size)

        for X_batch, Y_batch in s140te:

            if(add_dim_to_input == True):
                X_batch = X_batch[:,np.newaxis]

            loss,acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("test loss", loss),("test acc",acc)])

            testCounter = testCounter + 1
            testAccVal = testAccVal + acc 

            #Calculate confusion matrix
            y_true = Y_batch
            y_test = model.predict_on_batch(X_batch)
            y_test = [int(round(y)) for y in y_test]     

            conf_matrix = conf_matrix + confusion_matrix(y_true,y_test)

        print("\n")

        #Append epoch stats to respective lists
        test_accuracies.append((testAccVal/testCounter))
        test_confusions.append(conf_matrix.tolist())

        #Write statistics at the end of epoch to JSON 
        write_to_json(train_times, save_path, "_traintimes")        
        write_to_json(test_accuracies,save_path, "_accuracies")
        write_to_json(test_confusions,save_path, "_confusions")
        write_to_json(costs,save_path,"_costs")

def write_to_json(obj, path_base, path_decorator):
        """
        Writes (or overwrites) a JSON file located via a regular 
        combination of path base and path_decorator, dumping a JSON
        representation of obj to that file. Utility function for various 
        callback events.
        """
        # path decorator goes between filename and extension
        path_part, ext = os.path.splitext(path_base)
        full_path = "{}{}{}".format(path_part, path_decorator, ext)
        with open(full_path, "w") as f:
            json.dump(obj, f)

def word_embedding_input(embedding_type,batch_size=30,num_words=99):

    if(embedding_type == 'word2vec'):
        embedder = wv.WordVectorEmbedder('word2vec')
    else:
        embedder = wv.WordVectorEmbedder('glove')   

    '''(s140tr, s140te),(train_size, test_size) = batch_data.split_and_batch(
        data_loader=None,doclength=None,batch_size=batch_size,h5_path='imdb_split.hd5',
        transformer_fun=lambda x: embedder.embed_words_into_vectors(data_utils.tokenize(x),
            num_features=num_words),
        normalizer_fun=lambda x: data_utils.normalize(x,encoding=None,reverse=False),
        flatten=False) ''' 

    (s140tr, s140te),(train_size,test_size) = batch_data.split_data(
            None, 
            h5_path='sentiment140_split.hd5', 
            overwrite_previous=False,
            shuffle=True)

    s140_train_batch = BatchIterator(s140tr,
            normalizer_fun=lambda x: data_utils.normalize(x, 
                min_length=70,
                max_length=150, 
                truncate_left=True,
                encoding=None,
                reverse=True),

            transformer_fun=lambda x: embedder.embed_words_into_vectors(data_utils.tokenize(x),
            num_features=num_words),

            flatten=False, 
            batch_size=batch_size,
            auto_reset=True)

    s140_test_batch = BatchIterator(s140te,
            normalizer_fun=lambda x: data_utils.normalize(x, 
                min_length=70,
                max_length=150, 
                truncate_left=True,
                encoding=None,
                reverse=True),

            transformer_fun=lambda x: embedder.embed_words_into_vectors(data_utils.tokenize(x),
            num_features=num_words),

            flatten=False, 
            batch_size=batch_size,
            auto_reset=True)


    return (s140_train_batch, s140_test_batch),(train_size, test_size)

def char_input(batch_size=30,chars_reversed=True):

    #(s140tr, s140te),(train_size, test_size) = batch_data.split_and_batch(
    #    data_loader=None,doclength=None,batch_size=batch_size,h5_path='imdb_split.hd5')
    sentData = Sentiment140()

    (s140tr, s140te),(train_size,test_size) = batch_data.split_data(
            None, 
            h5_path='sentiment140_split.hd5', 
            overwrite_previous=False,
            shuffle=True)

    s140_train_batch = BatchIterator(s140tr,
            normalizer_fun=lambda x: data_utils.normalize(x, 
                min_length=70,
                max_length=150, 
                encoding=None,
                reverse= not chars_reversed),

            transformer_fun=data_utils.to_one_hot,

            flatten=False, 
            batch_size=batch_size,
            auto_reset=True)

    s140_test_batch = BatchIterator(s140te,
            normalizer_fun=lambda x: data_utils.normalize(x, 
                min_length=70,
                max_length=150, 
                encoding=None,
                reverse= not chars_reversed),

            transformer_fun=data_utils.to_one_hot,

            flatten=False, 
            batch_size=batch_size,
            auto_reset=True)

    return (s140_train_batch, s140_test_batch),(train_size, test_size)

def char_cnn():

    num_features = 67
    num_chars = 150

    print('Build char cnn model...')
  
    #Set Parameters for final fully connected layers 
    fully_connected = [1024,1024,1]
    
    model = Sequential()

    #Input = #alphabet x 150
    model.add(Convolution2D(256,num_features,7,input_shape=(1,num_features,num_chars)))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #Input = 48 x 256
    model.add(Convolution2D(256,1,7))
    
    #Input = 42 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 40 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 38 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 36 X 256
    model.add(Convolution2D(256,1,3))
    
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

def word_cnn(embedding_type,num_words):

    if(embedding_type=='word2vec'):
        num_features = 300
    else:   
        num_features = 200

    print('Build word cnn model...')
  
    #Set Parameters for final fully connected layers 
    fully_connected = [1024,1024,1]
    
    model = Sequential()

    #Input = #alphabet=#alphabet(200 or 300) x 50
    model.add(Convolution2D(256,num_words,7,input_shape=(1,num_words,num_features)))
    
    #Input = 44 x 256
    model.add(Convolution2D(256,1,7))
    
    #Input = 38 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 36 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 34 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 32 X 256
    model.add(Convolution2D(256,1,3))
    
    #30 x 256
    model.add(Flatten())

    #Fully Connected Layers

    #Input is 30*256=7680 Output is 1024 
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

def word_lstm(embedding_type, num_words):
    
    if(embedding_type=='word2vec'):
        num_features = 300
    else:   
        num_features = 200

    print('Build word lstm model...')
    
    model = Sequential()

    model.add(LSTM(num_features,input_shape=(num_words,num_features),inner_init='orthogonal',forget_bias_init='one',activation='tanh',
        inner_activation='hard_sigmoid',truncate_gradient=-1))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    return model

def char_lstm():

    num_features = 10
    num_chars = 150

    print('Build char lstm model...')
    
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

    '''batch_size = 128

    sentData = Sentiment140()

    (s140tr, s140te),(tr,te) = batch_data.split_data(DataIterator(sentData.load_data),
        h5_path='sentiment140_split.hd5',
        overwrite_previous=False,
        shuffle=True)    

    s140_train_batch = BatchIterator(s140tr,
            normalizer_fun=lambda x: data_utils.normalize(x, 
            min_length=70,
            max_length=150, 
            truncate_left=True,
            encoding=None,
            reverse=True),

            transformer_fun=lambda x: data_utils.to_one_hot(x),
            
            flatten=False, 
            batch_size=batch_size)

    text,label = s140_train_batch.next()
    print("Text shape", text.shape)
    print("Sent shape", label.shape)

    print(''.join(data_utils.from_one_hot(text[0])))
    '''
