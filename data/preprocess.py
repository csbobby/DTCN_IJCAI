#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pdb
from keras.preprocessing.sequence import pad_sequences
from datetime import *
from utils.TimeUnit import *
import pandas as pd
import copy
import sys
import scipy.spatial.distance
def preprocess(algorithm,X_train,y_train,timesteps=5,timestamps=[],identifiers=[],time_align=True,time_continuous=True,maxContextLengh=5,discrete_time_type='day',discrete_time_start_offset=2,discrete_time_unit=4,dual_time_align=False,**kwargs):
  if "DTCN" in algorithm:
    X_train,y_train,time_weight=reshape_data_time_align(X_train,y_train,timestamps=timestamps,identifiers=identifiers,time_continuous=time_continuous,maxContextLengh=maxContextLengh,maxTimeStep=timesteps,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,return_sequences=False,time_weight_mode = kwargs['time_weight_mode'])
    return X_train,y_train,time_weight
  if time_align:
    if dual_time_align:
      X_train_continue,y_train_continue,time_weight=reshape_data_time_align(X_train,y_train,timestamps=timestamps,identifiers=identifiers,time_continuous=True,maxContextLengh=maxContextLengh,maxTimeStep=timesteps,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,return_sequences=False)
      X_train_skip,y_train_skip,time_weight=reshape_data_time_align(X_train,y_train,timestamps=timestamps,identifiers=identifiers,time_continuous=False,maxContextLengh=maxContextLengh,maxTimeStep=timesteps,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,return_sequences=False)
      return X_train_continue,y_train_continue,X_train_skip,y_train_skip
  if algorithm=="S2S_LSTM":
    if time_align:
      X_train,y_train,time_weight=reshape_data_time_align(X_train,y_train,timestamps=timestamps,identifiers=identifiers,time_continuous=time_continuous,maxContextLengh=maxContextLengh,maxTimeStep=timesteps,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
      loss_mask = copy.deepcopy(y_train)
      for (x,y,z), value in np.ndenumerate(y_train):
          if value==0:
              loss_mask[x][y][z]=0.0
          else:
              loss_mask[x][y][z]=1.0

      loss_sample_weight = copy.deepcopy(y_train)
      for (x,y,z), value in np.ndenumerate(y_train):
          if value==0:
              loss_sample_weight[x][y][z]=0.0
          else:
              loss_sample_weight[x][y][z]=1.0
      loss_mask = np.array(loss_mask)
      loss_sample_weight = np.array(loss_sample_weight)
      return X_train,y_train,loss_mask,loss_sample_weight
    else:
      X_train,y_train = reshape_data('LSTM',X_train,y_train,timesteps,return_sequences=True)
      return X_train,y_train,np.array([]),np.array([])
  if ("LSTM" in algorithm) or ("GRU" in algorithm):
    if time_align:
      X_train,y_train,time_weight=reshape_data_time_align(X_train,y_train,timestamps=timestamps,identifiers=identifiers,time_continuous=time_continuous,maxContextLengh=maxContextLengh,maxTimeStep=timesteps,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,return_sequences=False)
      return X_train,y_train
    return reshape_data(algorithm,X_train,y_train,timesteps)
  return np.array(X_train,dtype=object),np.array(y_train,dtype=object)


def reshape_data(algorithm,X_train,y_train,nb_timesteps,return_sequences=False):
  reshaped_data=[]
  reshaped_y_train=[]
  if X_train.__class__==np.ndarray:
    X_train=X_train.tolist()
  end_index=-1
  for element in X_train:
    end_index+=1
    start_index=end_index+1-nb_timesteps
    if start_index<0:
      print "skip %d"%end_index
      continue
    slicing=X_train[start_index:end_index+1]
    if len(slicing)==0:
      raise Exception('ERROR IN PREPROCESS', 'RESHAPE')
    else:
      if algorithm=="CLSTM" or algorithm=="PLAIN_CLSTM" or algorithm=="CGRU":
        slice=np.average(np.array(slicing),axis=0).tolist()
        reshaped_data.append([slice,X_train[end_index]])
      elif algorithm=="LSTM" or algorithm=="PLAIN_LSTM" or algorithm=="GRU":
        reshaped_data.append(slicing)
      if len(y_train)>0:
        if return_sequences:
          pending = []
          for e in y_train[start_index:end_index+1]:
            pending.append([e])
          reshaped_y_train.append(pending)
        else:
          reshaped_y_train.append(float(y_train[end_index]))
  return np.array(reshaped_data,dtype=object),np.array(reshaped_y_train,dtype=object)



def reshape_data_time_align(X_train,y_train,timestamps=[],identifiers=[],time_continuous=True,maxContextLengh=3,maxTimeStep=20,discrete_time_type="hour",discrete_time_start_offset=2,discrete_time_unit=4,return_sequences=True,time_weight_mode = 'backup'):
  time_weight = []
  reshaped_data=[]
  reshaped_y_train=[]
  if X_train.__class__==np.ndarray:
    X_train=X_train.tolist()
  end_index=-1
  
  #开启了time alignment的时候：
  time_indexes=[]
  date_strings=[]
  for timestamp in timestamps:
    timestamp=int(timestamp)
    time_index=TimeUnit(datetime.datetime.fromtimestamp(timestamp),discrete_time_type,discrete_time_start_offset,discrete_time_unit).block_index()
    time_indexes.append(time_index)
    if discrete_time_type=="hour":
      date_strings.append("%s-%d"%(str(datetime.datetime.fromtimestamp(timestamp).date()),time_index))
    elif discrete_time_type=="day":
      date_strings.append("%s"%(str(datetime.datetime.fromtimestamp(timestamp).date())))
  data_index=np.arange(len(X_train))
  format_data=pd.DataFrame({"index":data_index,'time':np.array(timestamps),"identifier":np.array(identifiers),"time_index":np.array(time_indexes),"date_string":np.array(date_strings)})
  if len(y_train)==0:
      y_train=np.full(len(X_train),0,int)
  calculated_timesteps=0
  X_train=np.array(X_train)
  y_train=np.array(y_train)
  timestamps=np.array(timestamps)
  identifiers=np.array(identifiers)
  if time_continuous==True:
    maxRangeDay=maxContextLengh
    for element in X_train:
      end_index+=1
      #样本点的时间戳
      sample_timestamp=timestamps[end_index]
      #样本点的标识符（userid或者category）
      sample_identifier=identifiers[end_index]
      #找到时间起始点的时间戳，起始点=样本时间戳-maxRangeDay
      start_sample_timestamp=sample_timestamp-60*60*24*maxRangeDay
      #找到符合条件的bool结果
      bool_result=(timestamps<=sample_timestamp)&(timestamps>=start_sample_timestamp)&(identifiers==sample_identifier)
      find_result=format_data[bool_result].groupby('date_string').groups
      if len(format_data[bool_result])<1:
        continue
      slicing_X=[]
      slicing_Y=[]
      random_indexes = []
      for key in find_result.keys():
        random_index=np.random.choice(np.array(find_result[key]))
        random_indexes.append(random_index)
        slicing_X.append(X_train[random_index])
        slicing_Y.append(y_train[random_index])
      if len(slicing_X)>maxTimeStep:
        slicing_X=slicing_X[len(slicing_X)-maxTimeStep:]
        slicing_Y=slicing_Y[len(slicing_X)-maxTimeStep:]
      calculated_timesteps=max(calculated_timesteps,len(slicing_X))
      #将结果加入到最终结果
      reshaped_data.append(slicing_X)
      yslice=[]
      for y in slicing_Y:
          yslice.append([y])
      if return_sequences:
        reshaped_y_train.append(yslice)
      else:
        reshaped_y_train.append(slicing_Y[-1])
      #计算时间信息
      time_slice = timestamps[random_indexes]
      intervals = []

      for t_index in xrange(0,len(time_slice)-1):
        if time_weight_mode == 'backup':
          intervals.append(1/(np.exp((time_slice[t_index+1]-time_slice[t_index])/100000)))
        elif time_weight_mode == 'time_flag':
          datetime1 = datetime.datetime.fromtimestamp(time_slice[t_index+1])
          datetime0 = datetime.datetime.fromtimestamp(time_slice[t_index])
          date_list1 = [datetime1.day, datetime1.hour, datetime1.minute, datetime1.second]
          date_list0 = [datetime0.day, datetime0.hour, datetime0.minute, datetime0.second]
          sim = 1/scipy.spatial.distance.cosine(date_list1,date_list0)
          intervals.append(sim)
        else:
          intervals.append(time_slice[t_index+1]-time_slice[t_index]+1)
      intervals.append(1)
      intervals = np.exp(intervals)
      interval_sum = np.sum(intervals)
      for x in xrange(0,len(intervals)):
        intervals[x] /= interval_sum
      # intervals = pad_sequences(intervals,dtype='float',maxlen=maxTimeStep,value = sys.float_info.max)
      time_weight.append(intervals)
      #结束计算时间信息
    reshaped_data=pad_sequences(reshaped_data,dtype='float',maxlen=maxTimeStep)
    if return_sequences:
      reshaped_y_train=pad_sequences(reshaped_y_train,value=0,dtype='float',maxlen=maxTimeStep)    
  elif time_continuous==False:
    maxRangeDay=maxContextLengh*TimeUnit(datetime.datetime.fromtimestamp(0000000000),discrete_time_type).time_interval_between_unit_in_day()
    time_block_index_sequence=[]
    for timestamp in timestamps:
      time_block_index_sequence.append(TimeUnit(datetime.datetime.fromtimestamp(int(timestamp)),discrete_time_type,param_start_offset=discrete_time_start_offset,param_unit=discrete_time_unit).block_index())
    time_block_index_sequence=np.array(time_block_index_sequence)
    for element in X_train:
      
      end_index+=1
      #样本点的时间index
      sample_time_block_index=time_block_index_sequence[end_index]
      #样本点的时间戳
      sample_timestamp=timestamps[end_index]
      #样本点的标识符（userid或者category）
      sample_identifier=identifiers[end_index]
      #找到时间起始点的时间戳，起始点=样本时间戳-maxRangeDay
      start_sample_timestamp=sample_timestamp-60*60*24*maxRangeDay
      #找到符合条件的bool结果
      bool_result=(timestamps<=sample_timestamp)&(timestamps>=start_sample_timestamp)&(identifiers==sample_identifier)&(time_block_index_sequence==sample_time_block_index)
      #将bool结果转换成索引（index）
      find_result=format_data[bool_result].groupby('date_string').groups
      if len(format_data[bool_result])<1:
        continue
      slicing_X=[]
      slicing_Y=[]
      random_indexes = []
      for key in find_result.keys():
        random_index=np.random.choice(np.array(find_result[key]))
        random_indexes.append(random_index)
        slicing_X.append(X_train[random_index])
        slicing_Y.append(y_train[random_index])
      if len(slicing_X)>maxTimeStep:
        slicing_X=slicing_X[len(slicing_X)-maxTimeStep:]
        slicing_Y=slicing_Y[len(slicing_X)-maxTimeStep:]
      reshaped_data.append(slicing_X)
      
      yslice=[]
      for y in slicing_Y:
          yslice.append([y])
      if return_sequences:
        reshaped_y_train.append(yslice)
      else:
        reshaped_y_train.append(slicing_Y[-1])
      
      calculated_timesteps=max(calculated_timesteps,len(slicing_X))
      #计算时间信息
      time_slice = timestamps[random_indexes]
      intervals = []
      for t_index in xrange(0,len(time_slice)-1):
        if time_weight_mode == 'backup':
          intervals.append(1/(np.exp((time_slice[t_index+1]-time_slice[t_index])/100000)))
        elif time_weight_mode == 'time_flag':
          datetime1 = datetime.datetime.fromtimestamp(time_slice[t_index+1])
          datetime0 = datetime.datetime.fromtimestamp(time_slice[t_index])
          date_list1 = [datetime1.day, datetime1.hour, datetime1.minute, datetime1.second]
          date_list0 = [datetime0.day, datetime0.hour, datetime0.minute, datetime0.second]
          sim = 1/scipy.spatial.distance.cosine(date_list1,date_list0)
          intervals.append(sim)
        else:
          intervals.append(time_slice[t_index+1]-time_slice[t_index]+1)
      intervals.append(1)
      intervals = np.exp(intervals)
      interval_sum = np.sum(intervals)
      for x in xrange(0,len(intervals)):
        intervals[x] /= interval_sum
      # intervals = pad_sequences(intervals,dtype='float',maxlen=maxTimeStep,value = sys.float_info.max)
      time_weight.append(intervals)
      #结束计算时间信息
  reshaped_data=pad_sequences(reshaped_data,dtype='float',maxlen=maxTimeStep)
  time_weight = pad_sequences(time_weight,dtype='float',maxlen=maxTimeStep)
  if return_sequences:
    reshaped_y_train=pad_sequences(reshaped_y_train,value=0,dtype='float',maxlen=maxTimeStep)
  return reshaped_data,np.array(reshaped_y_train),np.array(time_weight)