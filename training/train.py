import torch
import wandb
from models import TModel, NormalModel, NormalModelGlobalSigma, TDistributionLoss, NormalDistributionLoss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from thop import profile

def get_optimizer(optimizer_name, parameters, lr):
    if optimizer_name == 'adam':
        return optim.Adam(parameters, lr=lr)
    elif optimizer_name == 'adamw':
        return optim.AdamW(parameters, lr=lr)
    elif optimizer_name == 'sgd':
        return optim.SGD(parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def count_flops(model, input_size):
    input_tensor = torch.randn(1, input_size).to(next(model.parameters()).device)
    flops, _ = profile(model, inputs=(input_tensor,))
    return flops
def count_flops(model, input_size):
    input_tensor = torch.randn(1, input_size).to(next(model.parameters()).device)
    flops, _ = profile(model, inputs=(input_tensor,))
    return flops

def train(config, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, device):
    input_size = X_train_tensor.shape[1]

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    models = {
        'TModel': TModel(input_size, config.hidden_size, config.num_hidden_layers, config.dropout_rate, config.activation_function).to(device),
        'NormalModel': NormalModel(input_size, config.hidden_size, config.num_hidden_layers, config.dropout_rate, config.activation_function).to(device),
        'NormalModelGlobalSigma': NormalModelGlobalSigma(input_size, config.hidden_size, config.num_hidden_layers, config.dropout_rate, config.activation_function).to(device)
    }

    criterions = {
        'TModel': TDistributionLoss(),
        'NormalModel': NormalDistributionLoss(),
        'NormalModelGlobalSigma': NormalDistributionLoss()
    }

    optimizers = {
        'TModel': get_optimizer(config.optimizer, models['TModel'].parameters(), config.learning_rate),
        'NormalModel': get_optimizer(config.optimizer, models['NormalModel'].parameters(), config.learning_rate),
        'NormalModelGlobalSigma': get_optimizer(config.optimizer, models['NormalModelGlobalSigma'].parameters(), config.learning_rate)
    }

    max_flops = config.max_flops if hasattr(config, 'max_flops') else 1e12
    flops_per_forward = {name: count_flops(model, input_size) for name, model in models.items()}
    total_flops = {name: 0 for name in models}

    best_test_loss = float('inf')
    best_model_name = None

    try:
        for epoch in range(config.num_epochs):
            # Training
            for name, model in models.items():
                model.train()

            train_losses = {name: 0 for name in models}
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                for name, model in models.items():
                    optimizers[name].zero_grad()
                    output = model(X_batch)
                    
                    if isinstance(output, tuple):
                        loss = criterions[name](*output, y_batch)
                    else:
                        loss = criterions[name](output, y_batch)

                    if torch.isnan(loss):
                        raise ValueError(f'{name}: Epoch {epoch+1}: Detected NaN loss, stopping training.')

                    loss.backward()
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            wandb.log({f"{name}_grad_norm": param.grad.norm().item()})
                            wandb.log({f"{name}_hist": wandb.Histogram(param.data.cpu())})
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizers[name].step()

                    train_losses[name] += loss.item()
                    total_flops[name] += flops_per_forward[name] * X_batch.size(0) * 2  # *2 for forward and backward pass

                    if total_flops[name] > max_flops:
                        raise StopIteration(f'{name}: Reached max FLOPs ({max_flops}), stopping training.')

            # Evaluation
            for name, model in models.items():
                model.eval()

            test_losses = {name: 0 for name in models}
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    for name, model in models.items():
                        output = model(X_batch)
                        
                        if isinstance(output, tuple):
                            loss = criterions[name](*output, y_batch)
                        else:
                            loss = criterions[name](output, y_batch)

                        if torch.isnan(loss):
                            raise ValueError(f'{name}: Epoch {epoch+1}: Detected NaN test loss, stopping training.')

                        test_losses[name] += loss.item()
                        total_flops[name] += flops_per_forward[name] * X_batch.size(0)

                        if total_flops[name] > max_flops:
                            raise StopIteration(f'{name}: Reached max FLOPs ({max_flops}), stopping training.')

            # Logging
            for name in models:
                avg_train_loss = train_losses[name] / len(train_loader)
                avg_test_loss = test_losses[name] / len(test_loader)
                
                wandb.log({
                    'flops': total_flops[name],
                    f'{name}_train_loss': avg_train_loss,
                    f'{name}_test_loss': avg_test_loss,
                })

                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_model_name = name

                if (epoch + 1) % 10 == 0 or epoch == config.num_epochs - 1:
                    print(f'{name} - Epoch {epoch+1}/{config.num_epochs}, '
                          f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
                          f'Total FLOPs: {total_flops[name]}')

    except StopIteration as e:
        print(str(e))
    except ValueError as e:
        print(str(e))
    except Exception as e:
        print(f"Training stopped due to an unexpected error: {str(e)}")
    finally:
        wandb.log({
            'best_model': best_model_name,
            'best_test_loss': best_test_loss,
            'total_flops': max(total_flops.values())
        })

    return models, best_model_name, best_test_loss