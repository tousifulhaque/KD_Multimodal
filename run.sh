#!/bin/bash

python3 processing.py --source '/Users/tousif/Lstm_transformer/KD_Multimodal/dataset/Labelled_Student_data/val' \
                        --dest '/Users/tousif/Lstm_transformer/KD_Multimodal/dataset/Labelled_Student_data/val/ADL' \
                        --data-type 'phone&watch'\
                        --types 'ADL'
