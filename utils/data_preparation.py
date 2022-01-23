import os
import pdb
import re
import numpy as np
import sklearn.decomposition

def read_file(filename,line_start,line_end,start=0, show_count=True,float_enable=True):
    data=[]
    with open(filename) as file_input:
        count=0
        for line in file_input:
            count+=1
            if show_count and count%10000==0:
                print "loading line %s"%count
            items=re.sub(r' +',' ',line.strip()).split(' ')[start:]
            l = []
            if (count<=line_start or count>line_end):
                l = 0
            elif len(items)==1:
                if float_enable:
                    l=float(float(items[0]))
                else:
                    l=((items[0]))
            else:
                for item in items:
                    try:
                        if float_enable:
                            l.append(float(item))
                        else:
                            l.append(item)
                    except Exception, e:
                        raise
                    else:
                        pass
                    finally:
                        pass
            data.append(l)
    return data

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def feature_loader(data_path,target_path,data_2_path="",part=11,iter=0,train_ratio=5,PCA_Enabled=False,float_enable=True):
    train_start=int((float(file_len(data_path))/float(part))*iter)
    train_end=int((float(file_len(data_path))/float(part))*(iter+train_ratio))
    test_start=train_end
    test_end = train_end+int((float(file_len(data_path))/float(part))*1)

    data=read_file(data_path,train_start,test_end,float_enable=float_enable)
    if len(data_2_path)>0:
        data=read_file(data_path,train_start,test_end,start=2,float_enable=float_enable)
        data2=read_file(data_2_path,train_start,test_end,start=1,float_enable=float_enable)
        data=np.column_stack((data,data2))
    label=np.full(len(data),0,dtype=int)
    if target_path!="":
        label=read_file(target_path,train_start,test_end,float_enable=float_enable)        
    if PCA_Enabled:
        print "PCA enabled"
        dims=len(data[0])
        pca=sklearn.decomposition.PCA()
        data=pca.fit_transform(data)
    assert len(data)==len(label), "Size of Training Set must equal to Size of Label!"
    
    X_train=data[train_start:train_end]
    y_train=label[train_start:train_end]
    X_test=data[test_start:test_end]
    y_test=label[test_start:test_end]
    del data
    del label
    datadir = os.path.split(data_path)[0]
    return datadir,X_train,y_train,X_test,y_test

