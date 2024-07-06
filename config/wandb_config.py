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
            'values': [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
        },
        'hidden_size': {
            'values': [16, 32, 64, 128, 256]
        },
        'num_hidden_layers': {
            'values': [1, 2, 3, 4]
        },
        'dropout_rate': {
            'values': [0, 0.1, 0.3, 0.5, .7, .9]
        },
        'num_epochs': {
            'values': [50, 100, 150]
        },
        'optimizer': {
            'values': ['adam', 'adamw', 'sgd']
        },
        'batch_size': {
            'values': [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        },
        'activation_function': {
            'values': ['relu', 'tanh', 'leaky_relu', 'elu', 'gelu', 'selu', 'swish', 'mish']
        }
    }
}
