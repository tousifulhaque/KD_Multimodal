"""
Run Time Configuration

Authors: Alexander Katrompas, Theodoros Ntakouris, Vangelis Metsis
Organization: Texas State University

"""

# ###############################
# command line parameter defaults
# ###############################
MODEL = 1 # set to 1 or 2
MODELS = [1,2]
VERBOSE = False
GRAPH = False

# ###############################
# data files
# ###############################
TRAIN = "dataset/new_watch_data_processed/watch_train.csv" # 
TEST = "new_watch_data_processed/watch_test.csv" # 
VALID =  "new_watch_data_processed/watch_val.csv" # optional, if none given test set will be used for validation

HEADER = 0
INDEX = 0

# ###############################
# Hyperparameters
# ###############################

# General
#BATCH_SIZE = 32
SEQLENGTH = 100
EPOCHS = 100
PATIENCE = 20
SHUFFLE = False

# LSTM
LSTM = 256
DENSE1 = 128
DENSE2 = 64
OUTPUT = 1
DROPOUT = 0.25
LSTM_ATTENTION = True

# Transformer
NUM_HEADS = 4
HEAD_SIZE = 128
FF_DIM = 16
NUM_ATTN_LAYERS = 4
MLP_DIMS = [16, 8] # can be []

#train config
config = {
        'epochs': 50,
        'length':50,
        'channel':3,
        'num_layers':  4,
        'embed_layer_size': 128,
        'global_clipnorm' : 3.0,
        'fc_layer_size': 16,
        'num_heads': 4,
        'dropout': 0.25,
        'attention_dropout': 0.25,
        'optimizer': 'adam',
        'amsgrad': False,
        'label_smoothing': 0.1,
        'learning_rate': 1e-3,
        #'weight_decay': {
        #    'values': [2.5e-4, 1e-4, 5e-5, 1e-5]
        'warmup_steps': 10,
        'batch_size': 16}
