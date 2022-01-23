'''
    Author: bobby
    Date created: Feb 1,2016
    Date last modified: May 14, 2016
    Python Version: 2.7
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
def IterationGraph(loss,val_loss,history_path,graph_out_path,features):
    if len(loss)==0 or len(val_loss)==0:
        return
    history_f = open(history_path,'w')
    length_history = len(loss)
    for i in range(length_history):
        if i<5:
            continue
        history_f.write("%s\t%s\n"%(str(loss[i]),str(val_loss[i])))
    history_f.close()
    df = pd.read_table(history_path,names=['loss','val_loss'])
    df=df.astype(float)
    loss_plot = df.plot(title='%s_training'%features)
    loss_fig = loss_plot.get_figure()
    loss_fig.savefig(graph_out_path)
    #loss_fig.savefig('USER_loss.png')






