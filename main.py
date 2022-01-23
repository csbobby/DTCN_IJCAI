import pandas as pd
import numpy as np
from memory_profiler import profile

from utils.IterationGraph import *
import sys
import datetime
import pdb
from utils.read_write_file import write_array_to_file
from utils.read_write_file import read_file_as_array
from utils.generate_h5 import check_h5, write_into_h5py, read_from_h5
import argparse
import os
print(os.environ['KERAS_BACKEND'])
import numpy as np
from utils.utils import cart_production, transform_info, flatten, generate_param_matrix


# argument processing #
parser = argparse.ArgumentParser()

# path relevant
parser.add_argument('-base_path', action='store',
    help='base path for all path', default="")
parser.add_argument('-feature_path', action='store',
    help='feature file path, usually visual feature')
parser.add_argument('-feature_test_path', action='store',
    help='feature file path, usually visual feature', default='')
parser.add_argument('-meta_path', action='store',
    help='meta file feature path, usually user feature', default='')
parser.add_argument('-label_path', action='store',
    help='label file path')
parser.add_argument('-label_test_path', action='store',
    help='label file path', default='')
parser.add_argument('-identifier_path', action='store', default='')
parser.add_argument('-timestamps_path', action='store', default='')
parser.add_argument('-h5_file_path', action='store',
    help='h5 file path to read/write', default=None)

# algorithm selection
parser.add_argument('-algorithm', action='store',nargs='?',
    help='algorithm to be used, default: LSTM', choices=['SHARED_DTCN','UNSHARED_DTCN','VANILLA_DTCN','PLAIN_LSTM','PLAIN_CLSTM','LSTM','GRU','CLSTM','CGRU','MLP','SVR',"S2S_LSTM"], default="LSTM")
# algorithm relevant parameters
parser.add_argument('-hidden_layer', action='store',
    help='(Optional LSTM/CLSTM ONLY!!!!) hidden_layer', default=64, type=int,nargs='?')
parser.add_argument('-timestep', action='store',
    help='(Optional LSTM/CLSTM ONLY!!!!) timestep', default=5, type=int,nargs='?')
parser.add_argument('-first_layer_output', action='store',
    help='(Optional MLP ONLY!!!!) first_layer_output', default=64, type=int,nargs='?')
parser.add_argument('-middle_layer_output', action='store',
    help='(Optional MLP ONLY!!!!) middle_layer_output', default=64, type=int,nargs='?')
parser.add_argument('-masking_enabled', action="store", choices=["y",'n'], default='n')
parser.add_argument('-visual_mlp_enabled',action="store",choices = ["y","n"], default='y')
parser.add_argument('-penalized_loss_enabled',action="store",choices = ["y","n"], default='n')
parser.add_argument('-penalized_loss_function',action="store",choices = ["sigmoid","tanh","relu","softplus","softsign","hard_sigmoid","linear","softmax"], default='sigmoid')
parser.add_argument('-time_align',action="store",choices = ["y","n"], default='n')
parser.add_argument('-time_dis_con',action="store",choices = ["discrete","continue"], default='continue')
parser.add_argument('-time_context_length',action="store", default=5)
parser.add_argument('-time_unit_metric',action="store",choices = ["day","hour"], default='day')
parser.add_argument('-discrete_time_start_offset',action="store", default=2)
parser.add_argument('-discrete_time_unit',action="store", default=4)
parser.add_argument('-merge_mode',action="store",choices = ["concat","mul","dot"], default='concat')
parser.add_argument('-lr',action="store", default=0.0001)
parser.add_argument('-dual_time_align', action="store", choices=["y",'n'], default='n')
parser.add_argument('-dual_lstm', help="dual lstm, olstm only!",action="store", choices=["y",'n'], default='n')
parser.add_argument('-model_path', help="model path for testing!",action="store", default='')
parser.add_argument('-time_weight_mode', help="Only For VANILLA_DTCN",action="store", choices=["backup",'time_flag'], default='backup')
parser.add_argument('-evaluation_metrics', help="all the evaluation metrics listed here",action="store", default="['spearman','mse','mae']")
parser.add_argument('-extra_info', help="Extra ",action="store", default='{}')

# experiment parameters
parser.add_argument('-data_split_part', action='store',
    help='(Optional) data_split_part', default=14, type=int,nargs='?')
parser.add_argument('-train_set_partial', action='store',
    help='(Optional) train_set_partial', default=9, type=int,nargs='?')
parser.add_argument('-total_cross_validation', action='store',
    help='(Optional ) total_cross_validation', default=5, type=int,nargs='?')
parser.add_argument('-start_cross_validation', action='store',
    help='(Optional ) start_cross_validation', default=0, type=int,nargs='?')
parser.add_argument('-exp_id', action='store',
    help='(Optional LSTM/CLSTM ONLY!!!!) exp_id', default='MYDATASET',nargs='?')
parser.add_argument('-nb_epoch', action='store',
    help='(Optional LSTM/CLSTM ONLY!!!!) nb_epoch', default=1000, type=int,nargs='?')
parser.add_argument('-batch_size', action='store',
    help='(Optional LSTM/CLSTM ONLY!!!!) batch_size', default=500, type=int,nargs='?')
parser.add_argument('-gpu_id', action='store',
    help='(Optional) gpu_id', default=-1,nargs='?')
# end of argument processing #

# argument extraction #
h5_file_path = parser.parse_args().h5_file_path
if len(parser.parse_args().feature_test_path) != 0:
    feature_test_path = parser.parse_args().base_path + os.path.sep + parser.parse_args().feature_test_path
else:
    feature_test_path = ''
if  len(parser.parse_args().label_test_path) != 0:
    label_test_path = parser.parse_args().base_path + os.path.sep + parser.parse_args().label_test_path
else:
    label_test_path = ''
time_weight_mode = parser.parse_args().time_weight_mode
model_path = parser.parse_args().model_path
dual_lstm=True
if parser.parse_args().dual_lstm=='n':
    dual_lstm=False
dual_time_align=True
if parser.parse_args().dual_time_align=='n':
    dual_time_align=False
lr = float(parser.parse_args().lr)
merge_mode = parser.parse_args().merge_mode
time_continuous=True
if parser.parse_args().time_dis_con=='discrete':
    time_continuous=False
maxContextLengh=int(parser.parse_args().time_context_length)
discrete_time_type=parser.parse_args().time_unit_metric
discrete_time_start_offset=int(parser.parse_args().discrete_time_start_offset)
discrete_time_unit=int(parser.parse_args().discrete_time_unit)
time_align = False
if parser.parse_args().time_align=='y':
    time_align=True

masking_enabled = False
if parser.parse_args().masking_enabled=='y':
    masking_enabled=True
algorithm = parser.parse_args().algorithm.upper()

feature_path = parser.parse_args().base_path + os.path.sep + parser.parse_args().feature_path

meta_path = parser.parse_args().base_path + os.path.sep + parser.parse_args().meta_path

label_path = parser.parse_args().base_path + os.path.sep + parser.parse_args().label_path
hidden_layer = parser.parse_args().hidden_layer
timestep = parser.parse_args().timestep
data_split_part = parser.parse_args().data_split_part
train_set_partial = parser.parse_args().train_set_partial
total_cross_validation = parser.parse_args().total_cross_validation
start_cross_validation = parser.parse_args().start_cross_validation
# exp_id = parser.parse_args().exp_id
original_exp_id = parser.parse_args().exp_id
MLP_ENABLED = True
Visual_MLP_ENABLED = False
if parser.parse_args().visual_mlp_enabled=='y':
    Visual_MLP_ENABLED=True
MLP_TAIL_ENABLED = False
nb_epoch = parser.parse_args().nb_epoch
batch_size = parser.parse_args().batch_size
gpu_id = parser.parse_args().gpu_id
first_layer_output = parser.parse_args().first_layer_output
middle_layer_output = parser.parse_args().middle_layer_output

identifier_path = parser.parse_args().base_path + os.path.sep + parser.parse_args().identifier_path
timestamps_path = parser.parse_args().base_path + os.path.sep + parser.parse_args().timestamps_path

if feature_path== parser.parse_args().base_path + os.path.sep:
    feature_path = ''
if meta_path== parser.parse_args().base_path + os.path.sep:
    meta_path = ''
if label_path== parser.parse_args().base_path + os.path.sep:
    label_path = ''
if identifier_path== parser.parse_args().base_path + os.path.sep:
    identifier_path = ''
if timestamps_path== parser.parse_args().base_path + os.path.sep:
    timestamps_path = ''


penalized_loss_enabled = False
if parser.parse_args().penalized_loss_enabled=='y':
    penalized_loss_enabled=True
    if not (algorithm=="LSTM" or algorithm=="CLSTM" or algorithm=="S2S_LSTM"):
        raise Exception("ONLY LSTM CAN USE PENALIZED LOSS FUNCTION!!!!","(LSTM ONLY NOW)")


penalized_loss_function = parser.parse_args().penalized_loss_function

import platform
if platform.system()=='Windows':
    if int(gpu_id)>=0:
        device_id = "gpu"+str(gpu_id)
        base_compiledir = os.path.join(os.environ['LOCALAPPDATA'], 'Theano', device_id)
        base_compiledir = base_compiledir.replace('\\','\\\\')
        os.environ['KERAS_BACKEND'] = 'theano'
        os.environ['THEANO_FLAGS'] = 'device=%s,lib.cnmem=1,base_compiledir=%s' % (device_id, base_compiledir)
elif int(gpu_id)>=0 and os.environ['KERAS_BACKEND']=='theano':
        import theano.sandbox.cuda
        theano.sandbox.cuda.use('gpu%s'%str(gpu_id))
# end of argument extraction #

from utils.training import training

from utils.testing import testing

from data.preprocess import preprocess



start_time=datetime.datetime.now()
config_data_path=feature_path
config_user_path=meta_path
target_path = label_path
import time
exp_id = "%s_%s_%s"%(algorithm,original_exp_id,datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

if meta_path=="":
    exp_id = "SINGLE_"+exp_id
import utils.data_preparation as dp
reload(sys)
sys.setdefaultencoding('utf8')
original_h5_file_path = h5_file_path

@profile
def run():
    if __name__ == "__main__":
        for i in xrange(start_cross_validation,total_cross_validation):
            datasets="%s_iter%d"%(exp_id,i)
            start_time=datetime.datetime.now()
            global h5_file_path
            if original_h5_file_path is not None:
                h5_file_path = original_h5_file_path.split('.')[0] + "{}_{}_{}".format(i, start_cross_validation, total_cross_validation) + original_h5_file_path.split('.')[1]
            print("H5 FILE PATH: {}".format(h5_file_path))
            ###################### load data ##########################
            print 'load data'
            if check_h5(h5_file_path) == False:
                data_path = config_data_path
                if model_path!='' and feature_test_path!='' and label_test_path!='':
                    X_train = []
                    y_train = []
                else:
                    datadir,X_train,y_train,X_test,y_test = dp.feature_loader(feature_path,target_path,part=data_split_part,iter=i,train_ratio=train_set_partial,PCA_Enabled=False)
                if feature_test_path!='' and label_test_path!='':
                    datadir,X_test,y_test,X_test_none,y_test_none = dp.feature_loader(feature_test_path,label_test_path,part=10,iter=0,train_ratio=10,PCA_Enabled=False)
                if len(meta_path)!=0:
                    datadir,X_train_visual,y_train,X_test_visual,y_test = dp.feature_loader(meta_path,target_path,part=data_split_part,iter=i,train_ratio=train_set_partial,PCA_Enabled=False)
                else:
                    X_train_visual=[]
                    X_test_visual=[]
                if algorithm=="S2S_LSTM" or time_align:
                    if len(identifier_path)==0 or len(timestamps_path)==0:
                        raise Exception("identifier_path/timestamps_path must be specified when enable time_align!")

                    unused,identifiers_train,unused,identifiers_test,unused = dp.feature_loader(identifier_path,target_path,part=data_split_part,iter=i,train_ratio=train_set_partial,PCA_Enabled=False,float_enable=False)
                    unused,timestamps_train,unused,timestamps_test,unused  = dp.feature_loader(timestamps_path,target_path,part=data_split_part,iter=i,train_ratio=train_set_partial,PCA_Enabled=False)
                else:
                    identifiers_train=[]
                    identifiers_test=[]
                    timestamps_train=[]
                    timestamps_test=[]

                #################preprocessing#########################
                X_train1 = []
                X_train2 = []
                X_train3 = []
                X_train4 = []
                X_test1 = []
                X_test2 = []
                X_test3 = []
                X_test4 = []
                loss_mask_train=[]
                loss_sample_weight_train=[]
                loss_mask_test=[]
                loss_sample_weight_test=[]
                time_weight_training = []
                time_weight_test = []
                print "preprocessing"

                if "DTCN" in algorithm:
                    X_train_visual,y_unused,time_weight_unused = preprocess(algorithm,X_train_visual,y_train,timestep,timestamps_train,identifiers_train,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,time_weight_mode = time_weight_mode)
                    if h5_file_path is not None:
                        write_into_h5py(h5_file_path, 'X_train_visual', X_train_visual)
                        del X_train_visual,y_unused,time_weight_unused
                    X_test_visual,y_unused,time_weight_unused = preprocess(algorithm,X_test_visual,y_test,timestep,timestamps_test,identifiers_test,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,time_weight_mode = time_weight_mode)
                    if h5_file_path is not None:
                        write_into_h5py(h5_file_path, 'X_test_visual', X_test_visual)
                        del y_unused,time_weight_unused
                    X_train,y_train,time_weight_training = preprocess(algorithm,X_train,y_train,timestep,timestamps_train,identifiers_train,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,time_weight_mode = time_weight_mode)
                    if h5_file_path is not None:
                        write_into_h5py(h5_file_path, 'X_train', X_train, 'y_train', y_train, 'time_weight_training', time_weight_training)
                        del X_train,y_train,time_weight_training
                    X_test,y_test,time_weight_test = preprocess(algorithm,X_test,y_test,timestep,timestamps_test,identifiers_test,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,time_weight_mode = time_weight_mode)
                    if h5_file_path is not None:
                        write_into_h5py(h5_file_path, 'X_test', X_test, 'y_test', y_test, 'time_weight_test', time_weight_test)
                        X_train, X_train_visual, y_train, loss_mask_train, loss_sample_weight_train, X_train1, X_train3, X_train2, X_train4, time_weight_training = [[],[],[],[],[],[],[],[],[],[]]

                elif dual_time_align:
                    X_train1,y_train1,X_train3,y_train1 = preprocess(algorithm,X_train_visual,y_train,timestep,timestamps_train,identifiers_train,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,dual_time_align=dual_time_align)
                    X_test1,y_test1,X_test3,y_test1 = preprocess(algorithm,X_test_visual,y_test,timestep,timestamps_test,identifiers_test,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,dual_time_align=dual_time_align)
                    X_train2,y_train1,X_train4,y_train1 = preprocess(algorithm,X_train,y_train,timestep,timestamps_train,identifiers_train,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,dual_time_align=dual_time_align)
                    X_test2,y_test1,X_test4,y_test1 = preprocess(algorithm,X_test,y_test,timestep,timestamps_test,identifiers_test,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit,dual_time_align=dual_time_align)
                    X_train = X_train1
                    X_train_visual=X_train3
                    y_train=y_train1
                    X_test=X_test1
                    X_test_visual=X_test3
                    y_test=y_test1
                elif algorithm=="S2S_LSTM":
                    X_train_visual,y_unused,uu,uu = preprocess(algorithm,X_train_visual,y_train,timestep,timestamps_train,identifiers_train,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
                    X_test_visual,y_unused,uu,uu = preprocess(algorithm,X_test_visual,y_test,timestep,timestamps_test,identifiers_test,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
                    X_train,y_train,loss_mask_train,loss_sample_weight_train = preprocess(algorithm,X_train,y_train,timestep,timestamps_train,identifiers_train,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
                    X_test,y_test,loss_mask_test,loss_sample_weight_test = preprocess(algorithm,X_test,y_test,timestep,timestamps_test,identifiers_test,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
                else:
                    X_train_visual,y_unused = preprocess(algorithm,X_train_visual,y_train,timestep,timestamps_train,identifiers_train,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
                    X_test_visual,y_unused = preprocess(algorithm,X_test_visual,y_test,timestep,timestamps_test,identifiers_test,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
                    X_train,y_train = preprocess(algorithm,X_train,y_train,timestep,timestamps_train,identifiers_train,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
                    X_test,y_test = preprocess(algorithm,X_test,y_test,timestep,timestamps_test,identifiers_test,time_align=time_align,time_continuous=time_continuous,maxContextLengh=maxContextLengh,discrete_time_type=discrete_time_type,discrete_time_start_offset=discrete_time_start_offset,discrete_time_unit=discrete_time_unit)
            else:
                X_train, X_train_visual, y_train, loss_mask_train, loss_sample_weight_train, X_train1, X_train3, X_train2, X_train4, time_weight_training = [[],[],[],[],[],[],[],[],[],[]]
                X_test = read_from_h5(h5_file_path, 'X_test')
                X_test_visual = read_from_h5(h5_file_path, 'X_test_visual')
                y_test = read_from_h5(h5_file_path, 'y_test')
                loss_mask_test = read_from_h5(h5_file_path, 'loss_mask_test')
                loss_sample_weight_test = read_from_h5(h5_file_path, 'loss_sample_weight_test')
                X_test1 = read_from_h5(h5_file_path, 'X_test1')
                X_test3 = read_from_h5(h5_file_path, 'X_test3')
                X_test2 = read_from_h5(h5_file_path, 'X_test2')
                X_test4 = read_from_h5(h5_file_path, 'X_test4')
                time_weight_test = read_from_h5(h5_file_path, 'time_weight_test')
            ################file write basic############################
            
            history_path = r'statics'+os.path.sep+'%s_history.txt'%(datasets)
            
            ################training###################################
            print 'training...'

            if model_path=='':
                model,loss,val_loss = training(algorithm,nb_epoch,batch_size,X_train,X_train_visual,y_train,
                                               datasets,hidden_layers=hidden_layer,mlp_enabled=MLP_ENABLED,
                                               mlp_tail=MLP_TAIL_ENABLED,timesteps=timestep,first_layer_output=first_layer_output,
                                               middle_layer_output=middle_layer_output,masking_enabled=masking_enabled,
                                               visual_mlp_enabled=Visual_MLP_ENABLED,penalized_loss_enabled=penalized_loss_enabled,
                                               penalized_loss_function=penalized_loss_function,loss_mask=loss_mask_train,
                                               loss_sample_weight=loss_sample_weight_train,lr=lr,merge_mode=merge_mode,
                                               dual_time_align=dual_time_align,X_train1=X_train1,X_train3=X_train3,X_train2=X_train2,
                                               X_train4=X_train4,dual_lstm=dual_lstm,time_weight_training=time_weight_training, h5_file_path=h5_file_path)
                ###############draw graph############################
                graph_out_path = r'statics'+os.path.sep+'Graph'+os.path.sep+'%s_loss.png'%(datasets)
                IterationGraph(loss,val_loss,history_path,graph_out_path,datasets)
                
                ###############testing###############################
            else:
                from keras.models import load_model
                model = load_model(model_path)

            if model==None:
                print "Previous Training has been done!"
                continue
            evaluation_results = testing(algorithm,model,X_test,X_test_visual,y_test,penalized_loss_enabled=penalized_loss_enabled,loss_mask=loss_mask_test,loss_sample_weight=loss_sample_weight_test,dual_time_align=dual_time_align,X_test1=X_test1,X_test3=X_test3,X_test2=X_test2,X_test4=X_test4,time_weight_test = time_weight_test,evaluation_metrics=evaluation_metrics)

            if model_path!='':
                continue
            end_time = datetime.datetime.now()
            print "time spent:",str(end_time-start_time)
            result=[config_data_path.split(os.path.sep)[-1],"iteration:",i,"timesteps:",timestep,"hidden layers:",hidden_layer,evaluation_results,str(end_time-start_time)]
            write_array_to_file('result'+os.path.sep+'%s.txt'%exp_id,[result])
            del X_train_visual
            del X_test_visual
        print "ALL DONE!"
    write_array_to_file('result'+os.path.sep+'%s.txt'%exp_id,[[str(parser.parse_args())],[str(extra_hints)]])



extra_hints = {}

import ast
evaluation_metrics = ast.literal_eval(parser.parse_args().evaluation_metrics)
extra_info = ast.literal_eval(parser.parse_args().extra_info)
param_set = generate_param_matrix(extra_info)
if len(param_set)==0:
    if os.environ['KERAS_BACKEND']=='tensorflow' and int(gpu_id)>=0:
        from keras import backend as K
        import tensorflow as tf
        with tf.device('/gpu:%s'%gpu_id):
            run()
    else:
        run()
else:
    for row in param_set:
        extra_hints = {}
        exp_id = "%s_%s_%s"%(algorithm,original_exp_id,datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        if meta_path=="":
            exp_id = "SINGLE_"+exp_id
        if row.has_key('feature_path'):
            extra_hints['feature_path'] = row['feature_path']
            feature_path = row['feature_path']
        if row.has_key('feature_test_path'):
            extra_hints['feature_test_path'] = row['feature_test_path']
            feature_test_path = row['feature_test_path']
        if row.has_key('meta_path'):
            extra_hints['meta_path'] = row['meta_path']
            meta_path = row['meta_path']
        if row.has_key('label_path'):
            extra_hints['label_path'] = row['label_path']
            label_path = row['label_path']
        if row.has_key('label_test_path'):
            extra_hints['label_test_path'] = row['label_test_path']
            label_test_path = row['label_test_path']
        if row.has_key('identifier_path'):
            extra_hints['identifier_path'] = row['identifier_path']
            identifier_path = row['identifier_path']
        if row.has_key('timestamps_path'):
            extra_hints['timestamps_path'] = row['timestamps_path']
            timestamps_path = row['timestamps_path']
        if row.has_key('algorithm'):
            extra_hints['algorithm'] = row['algorithm']
            algorithm = row['algorithm']
        if row.has_key('hidden_layer'):
            extra_hints['hidden_layer'] = row['hidden_layer']
            hidden_layer = row['hidden_layer']
        if row.has_key('timestep'):
            extra_hints['timestep'] = row['timestep']
            timestep = row['timestep']
        if row.has_key('data_split_part'):
            extra_hints['data_split_part'] = row['data_split_part']
            data_split_part = row['data_split_part']
        if row.has_key('train_set_partial'):
            extra_hints['train_set_partial'] = row['train_set_partial']
            train_set_partial = row['train_set_partial']
        if row.has_key('total_cross_validation'):
            extra_hints['total_cross_validation'] = row['total_cross_validation']
            total_cross_validation = row['total_cross_validation']
        if row.has_key('start_cross_validation'):
            extra_hints['start_cross_validation'] = row['start_cross_validation']
            start_cross_validation = row['start_cross_validation']
        if row.has_key('exp_id'):
            extra_hints['exp_id'] = row['exp_id']
            exp_id = row['exp_id']
        if row.has_key('nb_epoch'):
            extra_hints['nb_epoch'] = row['nb_epoch']
            nb_epoch = row['nb_epoch']
        if row.has_key('batch_size'):
            extra_hints['batch_size'] = row['batch_size']
            batch_size = row['batch_size']
        if row.has_key('gpu_id'):
            extra_hints['gpu_id'] = row['gpu_id']
            gpu_id = row['gpu_id']
        if row.has_key('first_layer_output'):
            extra_hints['first_layer_output'] = row['first_layer_output']
            first_layer_output = row['first_layer_output']
        if row.has_key('middle_layer_output'):
            extra_hints['middle_layer_output'] = row['middle_layer_output']
            middle_layer_output = row['middle_layer_output']
        if row.has_key('masking_enabled'):
            extra_hints['masking_enabled'] = row['masking_enabled']
            masking_enabled = row['masking_enabled']
        if row.has_key('visual_mlp_enabled'):
            extra_hints['visual_mlp_enabled'] = row['visual_mlp_enabled']
            visual_mlp_enabled = row['visual_mlp_enabled']
        if row.has_key('penalized_loss_enabled'):
            extra_hints['penalized_loss_enabled'] = row['penalized_loss_enabled']
            penalized_loss_enabled = row['penalized_loss_enabled']
        if row.has_key('penalized_loss_function'):
            extra_hints['penalized_loss_function'] = row['penalized_loss_function']
            penalized_loss_function = row['penalized_loss_function']
        if row.has_key('time_align'):
            extra_hints['time_align'] = row['time_align']
            time_align = row['time_align']
        if row.has_key('time_dis_con'):
            extra_hints['time_dis_con'] = row['time_dis_con']
            time_dis_con = row['time_dis_con']
        if row.has_key('time_context_length'):
            extra_hints['time_context_length'] = row['time_context_length']
            time_context_length = row['time_context_length']
        if row.has_key('time_unit_metric'):
            extra_hints['time_unit_metric'] = row['time_unit_metric']
            time_unit_metric = row['time_unit_metric']
        if row.has_key('discrete_time_start_offset'):
            extra_hints['discrete_time_start_offset'] = row['discrete_time_start_offset']
            discrete_time_start_offset = row['discrete_time_start_offset']
        if row.has_key('discrete_time_unit'):
            extra_hints['discrete_time_unit'] = row['discrete_time_unit']
            discrete_time_unit = row['discrete_time_unit']
        if row.has_key('merge_mode'):
            extra_hints['merge_mode'] = row['merge_mode']
            merge_mode = row['merge_mode']
        if row.has_key('lr'):
            extra_hints['lr'] = row['lr']
            lr = row['lr']
        if row.has_key('dual_time_align'):
            extra_hints['dual_time_align'] = row['dual_time_align']
            dual_time_align = row['dual_time_align']
        if row.has_key('dual_lstm'):
            extra_hints['dual_lstm'] = row['dual_lstm']
            dual_lstm = row['dual_lstm']
        if row.has_key('time_weight_mode'):
            extra_hints['time_weight_mode'] = row['time_weight_mode']
            time_weight_mode = row['time_weight_mode']
        if os.environ['KERAS_BACKEND']=='tensorflow' and int(gpu_id)>=0:
            from keras import backend as K
            import tensorflow as tf
            with tf.device('/gpu:%s'%gpu_id):
                run()
        else:
            run()
