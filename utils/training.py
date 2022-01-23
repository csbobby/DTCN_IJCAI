import numpy as np
import pdb
def training(algorithm,nb_epoch,batch_size,X_train,X_train_visual,y_train,datasets,datadir='.',hidden_layers=128,mlp_enabled=False,mlp_tail=False,timesteps=5,timestamps=[],first_layer_output=64, middle_layer_output=64,masking_enabled=False,visual_mlp_enabled=False,penalized_loss_enabled=False,penalized_loss_function = "sigmoid",loss_mask=[],loss_sample_weight=[],lr=0.0001,merge_mode='concat',dual_time_align=False,X_train1=[],X_train3=[],X_train2=[],X_train4=[],dual_lstm=False,**kwargs):
  gru_enabled=False
  if 'GRU' in algorithm:
    gru_enabled=True
  model=None
  loss=[]
  val_loss=[]
  if algorithm=="VANILLA_DTCN":
    from dtcn.vanilla_dtcn import lstm_training
    model,loss,val_loss = lstm_training(nb_epoch,batch_size,X_train,X_train_visual,y_train,datasets,time_weight_training = kwargs['time_weight_training'],hidden_layers=hidden_layers,mlp_enabled=mlp_enabled,mlp_tail=mlp_tail,masking_enabled=masking_enabled,visual_mlp_enabled=visual_mlp_enabled,lr=lr,merge_mode=merge_mode, h5_file_path= kwargs['h5_file_path'])
  elif algorithm=="SHARED_DTCN":
    from dtcn.main_shared_dtcn import lstm_training
    model,loss,val_loss = lstm_training(nb_epoch,batch_size,X_train,X_train_visual,y_train,datasets,time_weight_training = kwargs['time_weight_training'],hidden_layers=hidden_layers,mlp_enabled=mlp_enabled,mlp_tail=mlp_tail,masking_enabled=masking_enabled,visual_mlp_enabled=visual_mlp_enabled,lr=lr,merge_mode=merge_mode, h5_file_path= kwargs['h5_file_path'])
  elif algorithm=="UNSHARED_DTCN":
    from dtcn.unshared_dtcn import lstm_training
    model,loss,val_loss = lstm_training(nb_epoch,batch_size,X_train,X_train_visual,y_train,datasets,time_weight_training = kwargs['time_weight_training'],hidden_layers=hidden_layers,mlp_enabled=mlp_enabled,mlp_tail=mlp_tail,masking_enabled=masking_enabled,visual_mlp_enabled=visual_mlp_enabled,lr=lr,merge_mode=merge_mode, h5_file_path= kwargs['h5_file_path'])
  else:
    print "HAVEN'T IMPLEMENTED YET"
    raise Exception('HAVE NOT IMPLEMENTED YET, Exception!', '')

  return model,loss,val_loss

