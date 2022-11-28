sweep_config = {
  'method': 'grid',
  'metric': {
    'goal': 'maximize',
    'name': 'val_accuracy'
  },
  'parameters': {
      'epochs': {
        'value': 50
      },
      'num_layers': {
        'value': 3
      },
      'embed_layer_size': {
        'value': 128
      },
      'fc_layer_size': {
        'value': 256
      },
      'num_heads': {
        'value': 6
      },
      'dropout': {
        'value': 0.1
      },
      'attention_dropout': {
        'value': 0.1
      },
      'optimizer': {
        'value': 'adam'
      },
      'amsgrad': {
        'value': False
      },
      'label_smoothing': {
        'value': 0.1
      },
      'learning_rate': {
        'value': 1e-3
      },
      #'weight_decay': {
      #    'values': [2.5e-4, 1e-4, 5e-5, 1e-5]
      #},
      'warmup_steps': {
        'value': 5
      },
      'batch_size': {
        'value': 8
      },
      'global_clipnorm': {
        'value': 3.0
      },
    }
}