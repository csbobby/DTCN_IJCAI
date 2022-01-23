from keras.models import load_model
from utils.read_write_file import write_array_to_file
import utils.data_preparation as dp
from data.preprocess import preprocess
import argparse
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument('-feature_path', action='store')
parser.add_argument('-meta_path', action='store', default='')
parser.add_argument('-model_path', action='store')
parser.add_argument('-timestep', action='store',default=5,type=int)
parser.add_argument('-fout_path', action='store',default='')
parser.add_argument('-algorithm', action='store',nargs='?',
    help='algorithm to be used, default: LSTM', choices=['LSTM','GRU','CLSTM','CGRU','MLP','SVR'], default="LSTM")

feature_path = parser.parse_args().feature_path
meta_path = parser.parse_args().meta_path
model_path = parser.parse_args().model_path
timestep = parser.parse_args().timestep
fout_path = parser.parse_args().fout_path
algorithm = parser.parse_args().algorithm.upper()


model = load_model(model_path)
#####load data########
if len(meta_path)!=0:
    datadir,X_train_visual,y_train,X_test_visual,y_test = dp.feature_loader(meta_path,'',part=1,iter=0,train_ratio=1)
else:
    X_train_visual=[]
    X_test_visual=[]
unused,X_train,y_train,unused,unused = dp.feature_loader(feature_path,'',part=1,iter=0,train_ratio=1)
#####preprocess#######
X_train_visual,unused = preprocess(algorithm,X_train_visual,y_train,timestep)
X_train,y_train = preprocess(algorithm,X_train,y_train,timestep)
preds=[]
if (algorithm=="MLP" or algorithm=="SVR") and len(X_train_visual)>0:
    print "DUAL MLP/SVR TESTING ENABLED"
    X_train=np.column_stack((X_train,X_train_visual))
####predicting######
if len(X_train_visual)>0 and ("LSTM" in algorithm or "GRU" in algorithm):
    print "DUAL %s TESTING ENABLED"%(algorithm)
    preds = model.predict([X_train,X_train_visual], verbose=0)
else:
    preds = model.predict(X_train, verbose=0)
####print partial result#####
print preds[:200]
if fout_path!='':
  write_array_to_file(fout_path,preds)