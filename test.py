from transformer import transformer
from cfg import config, TEST
from utils import process_data

window_size = 50
stride = 5

model = transformer(length = config['length'],
       channels=config['channel'],
       num_heads=config['num_heads'],
       dropout_rate = config['dropout'],
       attention_dropout_rate = config['attention_dropout'],
       embed_dim =config['embed_layer_size'],
       mlp_dim = config['fc_layer_size'], 
       num_layers = config['num_layers'])

#load weight
weight_path = 'tmp/weights.ckpt'
model.load_weights(weight_path)

#processing test data
X_test, y_test = process_data(TEST,window_size,stride)

#evaluating model 
evaluation = model.evaluate(x=X_test,
    y=y_test,
    batch_size=config['batch_size'],
    verbose='auto',
    steps=len(X_test)/config['batch_size'],)

print("==========================")
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

