import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from transformer import transformer
from cfg import config, WINDOW,STRIDE, DATASET, TEST
from utils import process_data, F1_Score
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

model = transformer(length = config['length'],
       channels=config['channel'],
       num_heads=config['num_heads'],
       dropout_rate = config['dropout'],
       attention_dropout_rate = config['attention_dropout'],
       embed_dim =config['embed_layer_size'],
       mlp_dim = config['fc_layer_size'], 
       num_layers = config['num_layers'])

#load weight
weight_path = f'tmp/weights_{DATASET}_{WINDOW}.ckpt'
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
with open(f'transformer_{DATASET}_s{WINDOW}.tflite', 'wb') as f:
    f.write(tflite_model)

X_test, y_test = process_data(TEST, WINDOW, STRIDE)

#using interpreter to test
interpreter = tf.lite.Interpreter(model_path=f"transformer_{DATASET}_s{WINDOW}.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

data = X_test[1, :, :]
data = data[np.newaxis, :]

# Set input tensor to the interpreter
print(interpreter)
interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))
print('Before invoke')
# Run inference
interpreter.invoke()
print('After invoke')
# Get the output tensor and post-process the results (example)
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Inference result:", output_data)
