import tensorflow as tf
from utils.processing import process_data
from utils.imports import import_class
from train import compile_arguments


if  __name__ == "__main__":
    arg = compile_arguments()
    splited_dataset = process_data(**arg.dataset_args)
    X_test, y_test = splited_dataset['test']

    model = tf.keras.models.load_model('experiments/small(lstmattn)128_best/model/model_recall-0.73_precision-0.81.h5')
    model.summary()
    model.evaluate(x = X_test, y = y_test )
