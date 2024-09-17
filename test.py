import tensorflow as tf
from utils.processing import process_data
from utils.imports import import_class
from train import compile_arguments


if  __name__ == "__main__":
    arg = compile_arguments()
    splited_dataset = process_data(**arg.dataset_args)
    X_test, y_test = splited_dataset['val']

    model = tf.keras.models.load_model('experiments/smallrun256(conformer)_best/model/transformer_recall-0.89_precision-0.73.h5')
    model.evaluate(x = X_test, y = y_test )
