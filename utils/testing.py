from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras import objectives
import pandas as pd
from scipy import stats
import pdb
import numpy as np
# from keras import backend as K
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def testing(algorithm,model,X_test,X_test_visual,y_test, penalized_loss_enabled = False, loss_mask=[],loss_sample_weight=[],dual_time_align=False,X_test1=[],X_test3=[],X_test2=[],X_test4=[],evaluation_metrics=['spearman','mse','mae'],**kwargs):

    print("Evaluate the results...")
    preds=[]
    if "DTCN" in algorithm:
        preds = model.predict([X_test,X_test_visual,kwargs['time_weight_test']], verbose=0)
        objective_score = model.evaluate([X_test,X_test_visual,kwargs['time_weight_test']], y_test, batch_size=32)
    elif dual_time_align:
        preds = model.predict([X_test1,X_test2,X_test3,X_test4], verbose=0)
        if len(y_test.shape)==3:
            y_test = y_test[:,:,-1:].reshape(y_test.shape[0]*y_test.shape[1])
        objective_score = model.evaluate([X_test1,X_test2,X_test3,X_test4], y_test, batch_size=32)
    elif (algorithm=="MLP" or algorithm=="SVR" or algorithm=="PLAIN_LSTM" or algorithm=="PLAIN_CLSTM") and len(X_test_visual)>0:
        print "DUAL MLP/SVR/PLAIN_LSTM/PLAIN_CLSTM TESTING ENABLED"
        if "PLAIN_" in algorithm:
            X_test = np.concatenate((X_test,X_test_visual),axis=2)
        else:
            X_test=np.column_stack((X_test,X_test_visual))
        preds = model.predict(X_test)
        if algorithm!="SVR":
            objective_score = model.evaluate(X_test, y_test, batch_size=32)
        else:
            objective_score = "NOT EXIST IN SVR"
    elif len(X_test_visual)>0 and ("LSTM" in algorithm or "GRU" in algorithm):
        print "DUAL %s TESTING ENABLED"%(algorithm)
        
        if penalized_loss_enabled and algorithm=="S2S_LSTM":
            preds = model.predict([X_test,X_test_visual,loss_mask,loss_sample_weight], verbose=0)
            objective_score = model.evaluate([X_test,X_test_visual,loss_mask,loss_sample_weight], [y_test,y_test], batch_size=32)
        elif penalized_loss_enabled:
            preds = model.predict([X_test,X_test_visual], verbose=0)
            objective_score = model.evaluate([X_test,X_test_visual], [y_test,y_test], batch_size=32)
        else:
            preds = model.predict([X_test,X_test_visual], verbose=0)
            if len(y_test.shape)==3:
                y_test = y_test[:,:,-1:].reshape(y_test.shape[0]*y_test.shape[1])
            objective_score = model.evaluate([X_test,X_test_visual], y_test, batch_size=32)
    elif algorithm=="SVR":
        preds = model.predict(X_test)
        objective_score = "NOT EXIST IN SVR"
    else:
        preds = model.predict(X_test, verbose=0)
        objective_score = model.evaluate(X_test, y_test, batch_size=32)
    needed_preds=[]
    if "DTCN" in algorithm:
        for p in preds:
            needed_preds.append(p)
        needed_truths=[]
        for t in y_test:
            needed_truths.append(t)
    elif algorithm=="S2S_LSTM" and penalized_loss_enabled:
        for p in preds[0]:
            needed_preds.append(p[-1][-1])
        needed_truths=[]
        for t in y_test:
            needed_truths.append(t[-1][-1])
    elif algorithm=="S2S_LSTM":
        for p in preds:
            needed_preds.append(p[-1])
        needed_truths=[]
        for t in y_test:
            needed_truths.append(t[-1])
    elif penalized_loss_enabled:
        for p in preds[0]:
            needed_preds.append(p)
        needed_truths=[]
        for t in y_test:
            needed_truths.append(t)
    else:
        for p in preds:
            needed_preds.append(p)
        needed_truths=[]
        for t in y_test:
            needed_truths.append(t)
    y_list = []
    preds_list = []
    
    # pdb.set_trace()
    for i in range(len(needed_preds)):
        y_list.append(needed_truths[i])
        preds_list.append(needed_preds[i])

    y_list = np.array(y_list).flatten()
    preds_list = np.array(preds_list).flatten()
    print preds_list[:300]
    print y_list[:300]
    evaluation_results = {}
    if 'spearman' in evaluation_metrics:
        spearmanr_corr = stats.spearmanr(y_list, preds_list)
        print "Spearmanr Correlation",spearmanr_corr
        evaluation_results['spearman'] = spearmanr_corr[0]

    if 'mse' in evaluation_metrics:
        mse=mean_squared_error(y_list, preds_list)
        evaluation_results['mse'] = mse

    if 'mae' in evaluation_metrics:
        mae=mean_absolute_error(y_list, preds_list)
        evaluation_results['mae'] = mae

    if 'auc' in evaluation_metrics:
        evaluation_results['auc'] = auc_evaluation_metric(y_list,preds_list)

    if 'precision' in evaluation_metrics:
        evaluation_results['precision'] = precision_evaluation_metric(y_list,preds_list)

    if 'recall' in evaluation_metrics:
        evaluation_results['recall'] = recall_evaluation_metric(y_list,preds_list)

    for key in evaluation_results.keys():
        evaluation_results[key] = "%0.4f"%evaluation_results[key]
    print evaluation_results
    return evaluation_results


#BE CAUTION!!!!! USED IN XBOOST DATASET ONLY!

def auc_evaluation_metric(y_test,preds):
    from sklearn.metrics import roc_auc_score
    copyed_y_test = np.array(y_test).astype('int')
    copyed_preds = np.array(preds)
    auc = roc_auc_score(copyed_y_test,copyed_preds)
    return auc

def precision_evaluation_metric(y_test,preds):
    from sklearn.metrics import precision_score
    copyed_y_test = np.array(y_test).astype('int')
    threshold = sum(preds)*1.0/(1.0*len(preds))
    threshold_preds = thresholdcut(preds,threshold)
    precision = precision_score(copyed_y_test,threshold_preds)
    return precision


def recall_evaluation_metric(y_test,preds):
    from sklearn.metrics import recall_score
    copyed_y_test = np.array(y_test).astype('int')
    threshold = sum(preds)*1.0/(1.0*len(preds))
    threshold_preds = thresholdcut(preds.flatten(),threshold)
    recall = recall_score(copyed_y_test,threshold_preds)
    return recall



def thresholdcut(values,threshold):
    threshold_values = []
    for i in range(len(values)):
        if values[i] > threshold:
            threshold_values.append(1)
        else:
            threshold_values.append(0)
    return threshold_values