import tensorflow as tf
import numpy as np
import os

test_dataset_path = os.path.join(os.getcwd(), 'alex_test.npz')
test_data = np.load(test_dataset_path)
X_test = test_data['trials']
y_test = test_data['labels']
size = os.path.getsize('model.tflite')
print(size/(1024))
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
data = X_test[1, :, :]
data = data[np.newaxis, :]
print(type(data))
# Set input tensor to the interpreter
interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))
print('before invoke')
# Run inference
interpreter.invoke()
print('after invoke')
# Get the output tensor and post-process the results (example)
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Inference result:", output_data)