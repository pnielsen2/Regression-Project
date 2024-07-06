import torch
import torch.nn as nn
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

def log_model_stats(model, name):
    for param_name, param in model.named_parameters():
        wandb.log({
            f'{name}_{param_name}_mean': param.data.mean().item(),
            f'{name}_{param_name}_std': param.data.std().item(),
            f'{name}_{param_name}_max': param.data.max().item(),
            f'{name}_{param_name}_min': param.data.min().item(),
        })

def log_prediction_stats(model, X_test_tensor, name):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
        if isinstance(predictions, tuple):
            predictions = predictions[0]  # Assume first element is the mean prediction
        wandb.log({
            f'{name}_pred_mean': predictions.mean().item(),
            f'{name}_pred_std': predictions.std().item(),
            f'{name}_pred_max': predictions.max().item(),
            f'{name}_pred_min': predictions.min().item(),
        })

def log_loss_components(x, mu, sigma, nu=None):
    if nu is not None:  # T-distribution
        term1 = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
        term2 = -0.5 * torch.log(nu * torch.pi * sigma ** 2)
        term3 = -(nu + 1) / 2 * torch.log(1 + (1 / nu) * ((x - mu) / sigma) ** 2)
        wandb.log({
            'TModel_loss_term1': term1.mean().item(),
            'TModel_loss_term2': term2.mean().item(),
            'TModel_loss_term3': term3.mean().item(),
        })
    else:  # Normal distribution
        term1 = torch.log(sigma) + 0.5 * torch.log(torch.tensor(2 * torch.pi))
        term2 = 0.5 * ((x - mu) / sigma) ** 2
        wandb.log({
            'NormalModel_loss_term1': term1.mean().item(),
            'NormalModel_loss_term2': term2.mean().item(),
        })

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

    # Log data statistics
    wandb.log({
        'X_train_mean': X_train_tensor.mean().item(),
        'X_train_std': X_train_tensor.std().item(),
        'y_train_mean': y_train_tensor.mean().item(),
        'y_train_std': y_train_tensor.std().item(),
    })

    # Log model complexity
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        wandb.log({f'{name}_total_params': total_params})

    best_test_loss = float('inf')
    best_model_name = None

    try:
        for epoch in range(config.num_epochs):
            for name, model in models.items():
                model.train()

            train_losses = {name: 0 for name in models}
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                for name, model in models.items():
                    optimizers[name].zero_grad()
                    output = model(X_batch)
                    
                    if isinstance(model, TModel):
                        mu, sigma, nu = output
                        loss = criterions[name](y_batch, mu, sigma, nu)
                        if epoch == 0 and batch_idx < 5:
                            log_loss_components(y_batch, mu, sigma, nu)
                    else:
                        mu, sigma = output
                        loss = criterions[name](y_batch, mu, sigma)
                        if epoch == 0 and batch_idx < 5:
                            log_loss_components(y_batch, mu, sigma)

                    if torch.isnan(loss):
                        raise ValueError(f'{name}: Epoch {epoch+1}: Detected NaN loss, stopping training.')

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizers[name].step()

                    train_losses[name] += loss.item()
                    total_flops[name] += flops_per_forward[name] * X_batch.size(0) * 2  # *2 for forward and backward pass

                    if total_flops[name] > max_flops:
                        raise StopIteration(f'{name}: Reached max FLOPs ({max_flops}), stopping training.')

                    # Log learning rate and gradients
                    wandb.log({
                        f'{name}_lr': optimizers[name].param_groups[0]['lr'],
                        f'{name}_loss': loss.item(),
                        f'{name}_flops': total_flops[name]
                    })
                    for param_name, param in model.named_parameters():
                        if param.grad is not None:
                            wandb.log({f'{name}_{param_name}_grad_norm': param.grad.norm().item()})

                    # Log early iteration details
                    if epoch == 0 and batch_idx < 5:
                        wandb.log({
                            f'{name}_batch_{batch_idx}_input': wandb.Histogram(X_batch.cpu().numpy()),
                            f'{name}_batch_{batch_idx}_target': wandb.Histogram(y_batch.cpu().numpy()),
                            f'{name}_batch_{batch_idx}_output': wandb.Histogram(mu.detach().cpu().numpy()),
                        })

            # Evaluation
            for name, model in models.items():
                model.eval()

            test_losses = {name: 0 for name in models}
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    for name, model in models.items():
                        output = model(X_batch)
                        
                        if isinstance(model, TModel):
                            mu, sigma, nu = output
                            loss = criterions[name](y_batch, mu, sigma, nu)
                        else:
                            mu, sigma = output
                            loss = criterions[name](y_batch, mu, sigma)

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
                    f'{name}_train_loss': avg_train_loss,
                    f'{name}_test_loss': avg_test_loss,
                })

                if avg_test_loss < best_test_loss:
                    best_test_loss = avg_test_loss
                    best_model_name = name

                print(f'{name} - Epoch {epoch+1}/{config.num_epochs}, '
                      f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
                      f'Total FLOPs: {total_flops[name]}')

                # Log model and prediction stats
                log_model_stats(models[name], name)
                log_prediction_stats(models[name], X_test_tensor, name)

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