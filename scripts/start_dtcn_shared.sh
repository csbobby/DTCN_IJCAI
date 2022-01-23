DATA_HOME=/DATA/SMP
KERAS_BACKEND=theano \
THEANO_FLAGS='mode=FAST_RUN,device=cuda0,nvcc.fastmath=True,optimizer=fast_run' \
python main.py \
-feature_path $DATA_HOME/USER_20W_SORTED_BY_TIME.txt \
-meta_path $DATA_HOME/ResNet_20W_2048_SORTED_BY_TIME.txt \
-label_path $DATA_HOME/LABEL_20W_SORTED_BY_TIME.txt \
-algorithm SHARED_DTCN \
-nb_epoch 1000 \
-start_cross_validation 0 \
-total_cross_validation 5 \
-identifier_path $DATA_HOME/USERID_20W_SORTED_BY_TIME.txt \
-timestamps_path $DATA_HOME/TIMESTAMP_20W_SORTED_BY_TIME.txt \
-visual_mlp_enabled y \
-timestep 10 \
-time_align y \
-time_dis_con continue \
-time_context_length 18 \
-time_unit_metric hour \
-discrete_time_start_offset 2 \
-discrete_time_unit 4 \
-train_set_partial 9 \
-merge_mode concat \
-dual_time_align n \
-time_weight_mode time_flag \
-dual_lstm n