# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:55:08 2017

@author: v-fulon
"""
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def readSVMtoNumpy(test_p,ground,flag):
    count = 0
    f_test = open(test_p,'r')
    while 1:
        line_temp = f_test.readline()
        if not line_temp:
            break
        line_t = line_temp.split()
        if flag==1:
            ground.append(int(line_t[0]))
        else:
            ground.append(float(line_t[0]))
        #for i in range(1,len(line_t)):
        #    index_t = line_t[i].split(':')
        #    test_fea[count][int(index_t[0])-1] = float(index_t[1])
        count+=1
        if count % 50000 ==0:
            print('Test items: %d'% count)
    print('Test items: %d'% count)
    f_test.close()

def thresholdcut(pred_y,pred,threshold):
    for i in range(len(pred_y)):
        if pred_y[i] > threshold:
            pred.append(1)
        else:
            pred.append(0)
            

if __name__ == '__main__':
    
    test_n = 175335
    Dim = 275
    ground = []
    y_pred = []
    pred = []
    test_path = 'D:/users/v-fulon/work/R_project/data_new/svm_pos_neg/test_svm.txt'
    #result_path = 'D:/users/v-fulon/work/R_project/libFM/pred_libfm_3'
    result_path = 'D:/users/v-fulon/work/R_project/data_new/result/gbdt_onehot/test_leaf_pred.txt'
    
    print('Loading test data...')
    readSVMtoNumpy(test_path,ground,flag=1)
    readSVMtoNumpy(result_path,y_pred,flag=0)
    
    print('Start predicting test...')
    threshold = sum(y_pred)*1.0/(1.0*len(y_pred))
    thresholdcut(y_pred,pred,threshold)
    
    auc = roc_auc_score(ground,y_pred)
    precision = precision_score(ground,pred)
    recall = recall_score(ground,pred)
    print('auc: %f precision: %f recall: %f mean: %f'% (auc,precision,recall,threshold))
    
    
    
    