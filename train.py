import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from matplotlib import pyplot
from numpy import argmax

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Add, Dense, LayerNormalization,GlobalAveragePooling1D
from tensorflow.keras.metrics import Recall, Precision


#local imports
from transformer import transformer
from utils import process_data , F1_Score, cosine_schedule

#config
config = {
        'epochs': 50,
        'length':50,
        'channel':3,
        'num_layers':  2,
        'embed_layer_size': 16,
        'attention_head_dim' : 128,
        'global_clipnorm' : 3.0,
        'fc_layer_size': 32,
        'num_heads': 4,
        'dropout': 0.1,
        'attention_dropout': 0.0,
        'optimizer': 'adam',
        'amsgrad': False,
        'label_smoothing': 0.1,
        'learning_rate': 1e-3,
        #'weight_decay': {
        #    'values': [2.5e-4, 1e-4, 5e-5, 1e-5]
        'warmup_steps': 5,
        'batch_size': 32}
#creating model
if __name__ == '__main__':
    model = transformer(length = config['length'],
            channels=config['channel'],
            num_heads=config['num_heads'],
            dropout_rate = config['dropout'],
            attn_dim = config['attention_head_dim'],
            attention_dropout_rate = config['attention_dropout'],
            embed_dim =config['embed_layer_size'],
            mlp_dim = config['fc_layer_size'], 
            num_layers = config['num_layers'])


    #loading data
    train_dataset_path = os.path.join(os.getcwd(), 'new_watch_data_processed/watch_train.csv')
    val_dataset_path = os.path.join(os.getcwd(), 'new_watch_data_processed/watch_val.csv')
    window_size = 50
    stride = 5

    #processing train data 
    X_train, y_train = process_data(train_dataset_path, window_size, stride)
    #processing val data 
    X_val , y_val = process_data(val_dataset_path, window_size, stride)

    model.compile(
        loss= BinaryCrossentropy(label_smoothing=config['label_smoothing']),
        optimizer=Adam(
            global_clipnorm=config['global_clipnorm'],
            amsgrad=config['amsgrad'],
        ),
        metrics=[Recall(), Precision() , F1_Score()],
        )
    checkpoint_filepath = os.path.join(os.getcwd(), 'tmp/weights.ckpt')
    model_checkpoint = ModelCheckpoint(filepath = checkpoint_filepath, 
                                        save_weights_only = True, 
                                        monitor = 'val_f1_score', 
                                        mode = 'max', 
                                        save_best_only = True, 
                                        verbose = True)
    log_dir = "logs/"  # Specify the directory where TensorBoard logs will be saved
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(
        X_train,
        y_train,
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        validation_data=(X_val, y_val),
        shuffle = True,
        callbacks=[
            LearningRateScheduler(cosine_schedule(base_lr=config['learning_rate'], total_steps=config['epochs'], warmup_steps=config['warmup_steps'])),
            #EarlyStopping(monitor="loss", mode='min', min_delta=0.001, patience=5),
            model_checkpoint, tensorboard_callback
        ],
        verbose=1
        )
    
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