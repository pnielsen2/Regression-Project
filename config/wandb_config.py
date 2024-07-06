sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'best_test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'max_flops': {
            'value': 1e10
        },
        'learning_rate': {
            'values': [3e-4, 1e-4, 3e-5]
        },
        'hidden_size': {
            'values': [64, 128, 256, 512]
        },
        'num_hidden_layers': {
            'values': [1, 2, 3, 4, 5]
        },
        'dropout_rate': {
            'values': [0, 0.1, 0.3]
        },
        'num_epochs': {
            'value': 10000000
        },
        'optimizer': {
            'values': ['adam', 'adamw', 'sgd']
        },
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'activation_function': {
            'values': ['relu', 'tanh', 'leaky_relu', 'elu', 'gelu', 'selu', 'swish', 'mish']
        }
    }
}
