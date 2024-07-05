sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'best_test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'max_flops': {
            'values': [1e10, 1e11, 1e12]
        },
        'learning_rate': {
            'values': [0.0001, 0.001, 0.01]
        },
        'hidden_size': {
            'values': [64, 128, 256]
        },
        'num_hidden_layers': {
            'values': [2, 3, 4]
        },
        'dropout_rate': {
            'values': [0.1, 0.3, 0.5]
        },
        'num_epochs': {
            'values': [50, 100, 150]
        },
        'optimizer': {
            'values': ['adam', 'adamw', 'sgd']
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'activation_function': {
            'values': ['relu', 'tanh', 'leaky_relu', 'elu', 'gelu', 'selu', 'swish', 'mish']
        }
    }
}
