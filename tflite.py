import tensorflow as tf
import numpy as np
import os
from transformer import transformer
from train import config
from utils import process_data

window_size = 50
stride = 5
model = transformer(length = config['length'],
        channels=config['channel'],
        num_heads=config['num_heads'],
        dropout_rate = config['dropout'],
        attn_dim = config['attention_head_dim'],
        attention_dropout_rate = config['attention_dropout'],
        embed_dim =config['embed_layer_size'],
        mlp_dim = config['fc_layer_size'], 
        num_layers = config['num_layers'])

#load weight
weight_path = 'tmp/weights.ckpt'
model.load_weights(weight_path)

#convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

#adding special ops
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops. <-- Add this line
]
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops=True
converter.experimental_new_converter =True
converter.optimizations = [tf.lite.Optimize.DEFAULT]

#converting to tflite 
tflite_model = converter.convert()

#writing the tflite model to a file 
with open('transformer_tf24_s50.tflite', 'wb') as f:
    f.write(tflite_model)


test_dataset_path = os.path.join(os.getcwd(), 'new_watch_data_processed/watch_test.csv')
X_test, y_test = process_data(test_dataset_path, window_size, stride)

#using interpreter to test
interpreter = tf.lite.Interpreter(model_path="transformer_tf24_s50.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = X_test[1, :, :]
data = data[np.newaxis, :]

# Set input tensor to the interpreter
interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))

# Run inference
interpreter.invoke()

# Get the output tensor and post-process the results (example)
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Inference result:", output_data)