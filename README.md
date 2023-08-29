## Creating a conda environment 

```bash
# Code block
conda create -n tfold python=3.6.9
```


## Activating conda environment
```bash
conda activate tfold
````

## Installing Cuda Toolkit 
```bash
conda install -c conda-forge cudatoolkit=11.0
```
## Installing Cudnn 8.0
```bash
conda install -c "conda-forge/label/broken" cudnn
```
## Installing Tensorflow 2.4.0

```python
pip install tensorflow==2.4.0
```
## Installing Jupyter notebook 
```python
pip install notebook 
```
### Change in Code
Remove Normalization from the line below 
```python
from tensorflow.keras.layers import Add, Dense, LayerNormalization,Normalization,Masking, GlobalAveragePooling1D, Conv1D, Dropout, MultiHeadAttention, Layer
```
Add a new line 

```python 
from tensorflow.keras.layers.experimental.preprocessing import Normalization
```

Remove the keyword argument batch_size from the line below

```python
model.input_norm.adapt(X_train, batch_size=config['batch_size'])

