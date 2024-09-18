#!/bin/bash
#test
# python3 processing.py --source '/Users/tousif/Lstm_transformer/KD_Multimodal/dataset/Labelled_Student_data/val' \
#                         --dest '/Users/tousif/Lstm_transformer/KD_Multimodal/dataset/Labelled_Student_data/val/ADL' \
#                         --data-type 'phone&watch'\
#                         --types 'ADL'

CUDA_VISIBLE_DEVICES=3 python3 test.py --config 'config/lstmattn.yaml'
#CUDA_VISIBLE_DEVICES=3 python3 test.py --config 'config/conformer.yaml'
