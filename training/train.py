import torch
import wandb
from models import TModel, NormalModel, NormalModelGlobalSigma, TDistributionLoss, NormalDistributionLoss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
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

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs, max_flops):
    train_losses = []
    test_losses = []
    total_flops = 0
    flops_per_forward = count_flops(model, train_loader.dataset.tensors[0].shape[1])
    
    device = next(model.parameters()).device
    
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                
                if isinstance(model, nn.Module):  # Generic handling for any PyTorch model
                    output = model(X_batch)
                    if isinstance(output, tuple):
                        loss = criterion(*output, y_batch)
                    else:
                        loss = criterion(output, y_batch)
                else:
                    raise TypeError(f"Unsupported model type: {type(model)}")

                if torch.isnan(loss):
                    raise ValueError(f'Epoch {epoch+1}: Detected NaN loss, stopping training.')

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                total_flops += flops_per_forward * X_batch.size(0) * 2  # *2 for forward and backward pass
                
                if total_flops > max_flops:
                    raise StopIteration(f'Reached max FLOPs ({max_flops}), stopping training.')
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    if isinstance(model, nn.Module):
                        output = model(X_batch)
                        if isinstance(output, tuple):
                            batch_loss = criterion(*output, y_batch)
                        else:
                            batch_loss = criterion(output, y_batch)
                    else:
                        raise TypeError(f"Unsupported model type: {type(model)}")

                    if torch.isnan(batch_loss):
                        raise ValueError(f'Epoch {epoch+1}: Detected NaN test loss, stopping training.')

                    epoch_test_loss += batch_loss.item()
                    total_flops += flops_per_forward * X_batch.size(0)
                    
                    if total_flops > max_flops:
                        raise StopIteration(f'Reached max FLOPs ({max_flops}), stopping training.')
            
            avg_test_loss = epoch_test_loss / len(test_loader)
            test_losses.append(avg_test_loss)
            
            wandb.log({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'test_loss': avg_test_loss,
                'total_flops': total_flops
            })
            
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                print(f'{model.__class__.__name__} - Epoch {epoch+1}/{num_epochs}, '
                      f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
                      f'Total FLOPs: {total_flops}')

    except StopIteration as e:
        print(str(e))
    except ValueError as e:
        print(str(e))
    except Exception as e:
        print(f"Training stopped due to an unexpected error: {str(e)}")
    finally:
        return train_losses, test_losses, total_flops

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

    max_flops = config.max_flops if hasattr(config, 'max_flops') else 1e12  # Default to a large number if not specified

    best_test_loss = float('inf')
    best_model_name = None

    for name in models.keys():
        print(f'Training {name}...')
        train_losses, test_losses, total_flops = train_model(
            models[name], criterions[name], optimizers[name],
            train_loader, test_loader, config.num_epochs, max_flops
        )
        
        # Log the results, handling the case where no full epoch was completed
        if len(train_losses) > 0:
            for epoch in range(len(train_losses)):
                wandb.log({
                    f'{name}_train_loss': train_losses[epoch],
                    f'{name}_test_loss': test_losses[epoch] if epoch < len(test_losses) else None,
                    f'{name}_total_flops': total_flops,
                    f'{name}_flops_per_epoch': total_flops / (epoch + 1)
                })
            
            if test_losses and test_losses[-1] < best_test_loss:
                best_test_loss = test_losses[-1]
                best_model_name = name
        else:
            # Log that training failed to complete even one epoch
            wandb.log({
                f'{name}_training_failed': True,
                f'{name}_total_flops': total_flops
            })

    # Log the best model, handling the case where no model completed training
    if best_model_name:
        wandb.log({
            'best_model': best_model_name,
            'best_test_loss': best_test_loss,
            'total_flops': sum(wandb.run.summary.get(f'{name}_total_flops', 0) for name in models.keys())
        })
    else:
        wandb.log({
            'all_models_failed': True,
            'total_flops': sum(wandb.run.summary.get(f'{name}_total_flops', 0) for name in models.keys())
        })