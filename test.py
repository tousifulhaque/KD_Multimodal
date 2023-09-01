#environmental import 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Recall, Precision, Accuracy

from transformer import transformer
from cfg import config, TEST
from utils import process_data, F1_Score

window_size = 128
stride = 10
dataset = 'unimb'

model = transformer(length = config['length'],
       channels=config['channel'],
       num_heads=config['num_heads'],
       dropout_rate = config['dropout'],
       attention_dropout_rate = config['attention_dropout'],
       embed_dim =config['embed_layer_size'],
       mlp_dim = config['fc_layer_size'], 
       num_layers = config['num_layers'])

#load weight
weight_path =f'tmp/weights_{window_size}.ckpt'
model.load_weights(weight_path)
model.compile(
    loss= BinaryCrossentropy(label_smoothing=config['label_smoothing']),
    optimizer=Adam(
        global_clipnorm=config['global_clipnorm'],
        amsgrad=config['amsgrad'],
    ),
    metrics= [F1_Score(), Recall(), Precision(), Accuracy()],
    )

#processing test data
X_test, y_test = process_data(TEST,window_size,stride)

#evaluating model 
evaluation = model.evaluate(x=X_test,
    y=y_test,
    batch_size=config['batch_size'],
    verbose='auto',
    steps=len(X_test)/config['batch_size'],)

print("==========================")
print(evaluation)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

