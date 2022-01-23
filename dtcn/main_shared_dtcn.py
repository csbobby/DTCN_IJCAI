#!/usr/bin/python
# -*- coding: UTF-8 -*-
'''
    Author: bobby
    Date created: Feb 1,2016
    Date last modified: May 10, 2016
    Python Version: 2.7
'''
from keras.models import load_model
from keras.models import Model
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout,TimeDistributedDense,Flatten
from keras.layers import Input, LSTM, GRU, TimeDistributed, Masking
from keras.callbacks import ModelCheckpoint
from keras import callbacks as cb
import pandas as pd
import numpy as np
import pdb
from keras.optimizers import SGD,RMSprop
from keras.layers.pooling import AveragePooling1D
from keras.layers import Merge
import datetime
import sys
import keras.backend as K
from utils.training_log import *
from loss_function import *
from MTLSTM import MTLSTM
sys.setrecursionlimit(10000)

# MLP head:
# dense   dense
# dense<->dense
#     merge
#     MTLSTM

model_path=""
param=[]
initial_epoch=0
initial_losses=[]
initial_val_losses=[]
def lstm_training(nb_epoch,batch_size,X_train,X_train_visual,y_train,features,time_weight_training,datadir='.',hidden_layers=128,mlp_enabled=False,mlp_tail=False,timesteps=5,timestamps=[],masking_enabled=False,gru_enabled=False,visual_mlp_enabled = False, penalized_loss_enabled = False,penalized_loss_function = "sigmoid",lr=0.0001,merge_mode='concat', h5_file_path=None):
    global model_path,param,initial_epoch,initial_losses,initial_val_losses
    if len(y_train.shape)==3 and y_train.shape[2]>1:
        y_train = y_train[:,:,-1:].reshape(y_train.shape[0]*y_train.shape[1])
    model_path=""
    param=[]
    initial_epoch=0
    initial_losses=[]
    initial_val_losses=[]
    timesteps = len(X_train[0])
    nb_timesteps=timesteps

    param={}
    param['nb_epoch']=nb_epoch
    param['batch_size']=batch_size
    param['validation_split']=0.1

    # Model
    time_input = Input(batch_shape=(None, len(time_weight_training[0])))
    user_input = Input(batch_shape=(None, timesteps, len(X_train[0][0])))
    shared_time_dense4 = TimeDistributedDense(16)
    shared_activation4 = Activation('tanh')
    if masking_enabled:
        masking1 = Masking(mask_value=0,input_shape=(timesteps, len(X_train[0][0])))(user_input)
        time_dense1 = TimeDistributedDense(32,input_length=timesteps,input_dim=len(X_train[0][0]))(masking1)
    else:
        time_dense1 = TimeDistributedDense(32,input_length=timesteps,input_dim=len(X_train[0][0]))(user_input)
    user_output = time_dense1 
    if mlp_enabled==True:
        activation1 = Activation('tanh')(time_dense1)
        dropout1 = Dropout(0.5)(activation1)
        user_time_dense1 = shared_time_dense4(dropout1)
        user_activition1 = shared_activation4(user_time_dense1)
        user_output = user_activition1

    visual_input = Input(batch_shape=(None, timesteps, len(X_train_visual[0][0])))
    if masking_enabled:
        masking2 = Masking(mask_value=0,input_shape=(timesteps, len(X_train_visual[0][0])))(visual_input)
        time_dense3 = TimeDistributedDense(32,input_length=timesteps,input_dim=len(X_train_visual[0][0]))(masking2)
    else:
        time_dense3 = TimeDistributedDense(32,input_length=timesteps,input_dim=len(X_train_visual[0][0]))(visual_input)
    visual_output = time_dense3
    if visual_mlp_enabled==True:
        print "=============>visual mlp enabled"
        activation3 = Activation('tanh')(time_dense3)
        dropout2 = Dropout(0.5)(activation3)
        visual_time_dense1 = shared_time_dense4(dropout2)
        visual_activition1 = shared_activation4(visual_time_dense1)
        visual_output = visual_activition1
    merge1 = Merge(mode=merge_mode)([user_output,visual_output])
    if gru_enabled:
        raise Exception('GRU','NOT IMPLEMENTED')
        rnn1 = GRU(consume_less="gpu",output_dim = hidden_layers,input_length=timesteps,return_sequences=False)(merge1)
        # rnn2 = GRU(consume_less="gpu",output_dim = hidden_layers,input_length=timesteps,return_sequences=True)(merge1)
    else:
        rnn1 = MTLSTM(time_weight = time_input, consume_less="gpu",output_dim = hidden_layers,input_length=timesteps,return_sequences=False)(merge1)
        # rnn2 = LSTM(consume_less="gpu",output_dim = hidden_layers,input_length=timesteps,return_sequences=True)(merge1)
    rnn_output = rnn1
    if mlp_tail==True:
        dense_tail = Dense(32)(rnn1)
        activation_tail = Activation('tanh')(dense_tail)
        drop_tail = Dropout(0.5)(activation_tail)
        dense_tail1 = Dense(16)(drop_tail)
        activation_tail1 = Activation('tanh')(dense_tail1)
        rnn_output = activation_tail1
    dropout_main = Dropout(0.5)(rnn_output)
    dense_main = Dense(output_dim=1)(dropout_main)

    # Model Training
    if penalized_loss_enabled:
        loss_output = Dense(1,activation=penalized_loss_function)(rnn1)
        model = Model(input=[user_input,visual_input,time_input], output=[dense_main,loss_output])
    else:
        model = Model(input=[user_input,visual_input,time_input], output=dense_main)
    rmsprop=RMSprop(lr=lr, rho=0.9, epsilon=1e-08)
    if penalized_loss_enabled:
        print "=============>penalized loss enabled!!!!!(using %s)"%penalized_loss_function
        model.compile(loss=[normal_mean_squared_error(noise=dense_main),penalized_loss(noise=loss_output)],optimizer=rmsprop)
    else:
        model.compile(loss="mse",optimizer=rmsprop)

    # Model Resume
    import os
    model_path ="model"+os.path.sep+"%s_%s_%s"%(str(features),str(timesteps),str(hidden_layers))+".hdf5"
    #if the evalidation error decreased after one epoch, save the model
    checkpointer = cb.ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True,save_weights_only=True)

    # define a callback object
    history = LossHistory()

    if judge_finished(model_path)==False and get_epoch_num(model_path)>0:
        param['nb_epoch']-=get_epoch_num(model_path)
        model = load_model(model_path)
        initial_epoch=get_epoch_num(model_path)
        initial_losses=get_loss(model_path)
        initial_val_losses=get_val_loss(model_path)
    earlyStopping=cb.EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto')
    print("start train process...")

    if judge_finished(model_path)==True:
        initial_losses=get_loss(model_path)
        initial_val_losses=get_val_loss(model_path)
        return None,initial_losses,initial_val_losses

    if penalized_loss_enabled:
        hist = model.fit([X_train,X_train_visual,time_weight_training], [y_train,y_train],nb_epoch=param['nb_epoch'], batch_size=param['batch_size'], \
            validation_split=param.get('validation_split'), verbose=0 \
              , callbacks=[checkpointer,history,earlyStopping]
              )
    else:
        hist = model.fit([X_train,X_train_visual,time_weight_training], y_train,nb_epoch=param['nb_epoch'], batch_size=param['batch_size'], \
            validation_split=param.get('validation_split'), verbose=0 \
              , callbacks=[checkpointer,history,earlyStopping]
              )

    loss = get_loss(model_path)
    val_loss = get_val_loss(model_path)

    return model,loss,val_loss


# the callback function for logging loss
class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
    def on_epoch_end(self, epoch, logs={}):
        global model_path,param,initial_epoch,initial_losses,initial_val_losses

        if (True):
            update_status(model_path,epoch+initial_epoch,"False")
            print("%s: Actual Epoch %d"%(str(datetime.datetime.now()),epoch+initial_epoch))
            sys.stdout.flush()
            initial_losses.append(logs.get('loss'))
            initial_val_losses.append(logs.get('val_loss'))
            save_loss(model_path,initial_losses)
            save_val_loss(model_path,initial_val_losses)
    def on_train_end(self, logs={}):
        global model_path,param
        epoch1=get_epoch_num(model_path)
        update_status(model_path,epoch1,"True")