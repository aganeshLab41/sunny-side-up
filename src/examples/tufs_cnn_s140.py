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
from datasets.sentiment140 import Sentiment140

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D, MaxPooling2D, Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils,generic_utils
from sklearn.metrics import confusion_matrix

'''
Model below is based on paper by Xiang Zhang "Character-Level Convolutional
Networks for Text Classification" (http://arxiv.org/abs/1509.01626) paper was
formerly known as "Text Understanding from Scratch" (http://arxiv.org/pdf/1502.01710v4.pdf) 

THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32 python tufs_cnn_s140.py
'''
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

def model_defn():

    print('Build model...')
  
    #Set Parameters for final fully connected layers 
    fully_connected = [1024,1024,1]
    
    model = Sequential()

    #Input = #alphabet x 1014
    model.add(Convolution2D(256,67,7,input_shape=(1,67,150)))
    model.add(MaxPooling2D(pool_size=(1,3)))

    #Input = 336 x 256
    model.add(Convolution2D(256,1,7))
    #model.add(MaxPooling2D(pool_size=(1,3)))

    #Input = 110 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 108 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 106 x 256
    model.add(Convolution2D(256,1,3))

    #Input = 104 X 256
    model.add(Convolution2D(256,1,3))
    #model.add(MaxPooling2D(pool_size=(1,3)))
    #model.add(Convolution1D(256,7))

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

if __name__=="__main__":

    #Set batch size for input data
    batch_size = 128

    #Set the number of epochs to run
    num_epochs = 10

    #Import model 
    #model,sgd = model_defn()

    # Get training and testing sets, and their sizes for the amazon dataset
    # from HDF5 file that uses an 80/20 train/test split 
    (amtr,amte),(amntr,amnte) = datasets, sizes = batch_data.split_data(None,
        h5_path='sentiment140_split.hd5', overwrite_previous=False,shuffle=True)
    #sentData = Sentiment140()

    #Test not using HDF5 and see if behavior changes
    #(sentr,sendev,sente) = data_utils.split_data(sentData.load_data(return_iter=False),train=.8,dev=0,test=.2)

    #print("Sentiment140 tr shape", len(sentr))
    #print("First elem", sentr[0])
    #sentr = iter(sentr)
    #sente = iter(sente)
    

    #Generator that outputs S140 training data in batches with specificed parameters
    am_train_batch = batch_data.batch_data(amtr,normalizer_fun=lambda x: x,
        transformer_fun=lambda x: data_utils.to_one_hot(x),
        flatten=False, batch_size=batch_size)

    #Generator that outputs S140 testing data in batches with specificed parameters
    am_test_batch = batch_data.batch_data(amte,normalizer_fun=lambda x: x,
        transformer_fun=lambda x: data_utils.to_one_hot(x),
        flatten=False, batch_size=batch_size)

    tweet, sentiment = am_train_batch.next()
    print("Tweet batch shape",tweet.shape)
    print("Sentiment batch shape",sentiment.shape)

    counter = 0

    for e in range(2):
        for X_batch, Y_batch in am_train_batch:
            if counter % 100 == 0:
                print("Counter val",counter)
                print("epoch", e)
            counter = counter + 1    

    #print(tweet[0,0])
    #print(tweet[0][0])

    #oh = data_utils.to_one_hot(tweet[0,0])
    #print(np.array_str(np.argmax(oh,axis=0)))
    #print("Translated back into characters:\n")
    #print(''.join(data_utils.from_one_hot(oh)))


    '''
    #Statistics Initialization 
    train_times = []
    test_accuracies = []
    test_confusions = []
    costs = []

    save_path = 'tufs_s140.json'

    #Begin runs of training and testing    
    for e in range(num_epochs):
        print('-'*10)
        print('Epoch', e)
        print('-'*10)
        print("Training...")

        #Temporary stopping condition variable 
        #stop_count = 0

        #Initialize variables for collecting minibatch cost stats and starting train timer
        costs.append([])
        train_times.append({})
        train_times[e]['start'+'_epoch_'+str(e)] = time.time()

        #Halve the learning rate every 3 epochs 
        if(e % 3 == 2):
            sgd.lr.set_value(sgd.lr.get_value()/2)
        
        #Progress bar initialized for training data
        progbar = generic_utils.Progbar(amntr)

        #Training loop performing training process

        for X_batch, Y_batch in am_train_batch:
            
            #Temporary stopping condition
            #if(stop_count > 10):
            #    break

            #Reshape input from a 3D Input to a 4D input for training    
            X_batch = X_batch[:,np.newaxis]
            
            loss,acc = model.train_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("train loss", loss),("train acc",acc)])

            #Record costs after each minibatch
            costs[e].append(float(loss))
            
            #Temporary stopping condition variable
            #stop_count = stop_count + 1

        print("\n")    

        #Record how long training took per epoch
        train_times[e]['end'+'_epoch_'+str(e)] = time.time()
        
        #Save the model every epoch into hd5 file    
        model.save_weights('tufs_keras_weights.hd5',overwrite=True) 

        print("\nTesting...")

        #Progress bar initialized for testing data
        progbar = generic_utils.Progbar(amnte)

        #Temporary stopping condition var
        #stop_count = 0

        #Initialize variables for collecting test related stats
        testCounter = 0
        testAccVal = 0

        #Confusion matrix initializations
        y_true = []
        y_test = []
        conf_matrix = np.zeros((2,2))

        #Testing loop performing tests/predict based on model

        for X_batch, Y_batch in am_test_batch:
            
            #Temporary stopping condition
            #if(stop_count > 10):
            #    break
            
            #Reshape input from a 3D Input to a 4D input for training
            X_batch = X_batch[:,np.newaxis]

            loss,acc = model.test_on_batch(X_batch, Y_batch, accuracy=True)
            progbar.add(batch_size, values=[("test loss", loss),("test acc",acc)])

            #Temporary stopping condition variable
            #stop_count = stop_count + 1

            testCounter = testCounter + 1
            testAccVal = testAccVal + acc 

            #Calculate confusion matrix
            y_true = Y_batch
            y_test = model.predict_on_batch(X_batch)
            y_test = [int(round(y)) for y in y_test]     

            conf_matrix = conf_matrix + confusion_matrix(y_true,y_test)

        print("\n")

        print("Test counter",testCounter)
        print("Test acc value",testAccVal)
        #print("accuracies", float(testAccVal)/testCounter)

        #Temporary stopping condition variable
        #stop_count = 0

        #Append epoch stats to respective lists
        #test_accuracies.append((float(testAccVal)/testCounter))
        test_confusions.append(conf_matrix.tolist())


        #Write statistics at the end of epoch to JSON 
        write_to_json(train_times, save_path, "_traintimes")        
        #write_to_json(test_accuracies,save_path, "_accuracies")
        write_to_json(test_confusions,save_path, "_confusions")
        write_to_json(costs,save_path,"_costs")
    '''    

