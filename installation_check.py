import tensorflow as tf
print(f'Cudnn verion : {tf.sysconfig.get_build_info()["cudnn_version"]}')
print(tf.config.list_physical_devices('GPU'))