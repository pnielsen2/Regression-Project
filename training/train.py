import torch
import wandb
from models import TModel, NormalModel, NormalModelGlobalSigma, TDistributionLoss, NormalDistributionLoss
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from thop import profile
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import io
from PIL import Image

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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizers[name].step()

                    train_losses[name] += loss.item()
                    total_flops[name] += flops_per_forward[name] * X_batch.size(0) * 2  # *2 for forward and backward pass

                    if total_flops[name] > max_flops:
                        raise StopIteration(f'{name}: Reached max FLOPs ({max_flops}), stopping training.')

            # Evaluation
            test_losses = {name: 0 for name in models}
            output_params = {name: [] for name in models}
            
            for name, model in models.items():
                model.eval()
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    
                    for name, model in models.items():
                        if name == 'TModel':
                            mu, sigma, nu = model(X_batch)
                            loss = criterions[name](y_batch, mu, sigma, nu)
                            output_params[name].append((mu.cpu(), sigma.cpu(), nu.cpu()))
                        elif name == 'NormalModel':
                            mu, sigma = model(X_batch)
                            loss = criterions[name](y_batch, mu, sigma)
                            output_params[name].append((mu.cpu(), sigma.cpu()))
                        else:  # NormalModelGlobalSigma
                            mu, sigma = model(X_batch)
                            loss = criterions[name](y_batch, mu, sigma)
                            output_params[name].append((mu.cpu(), sigma.cpu()))

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
                
                # Calculate mean and std of output parameters
                if name == 'TModel':
                    mu_cat = torch.cat([batch[0] for batch in output_params[name]])
                    sigma_cat = torch.cat([batch[1] for batch in output_params[name]])
                    nu_cat = torch.cat([batch[2] for batch in output_params[name]])
                    output_stats = {
                        f'{name}_mu_mean': mu_cat.mean().item(),
                        f'{name}_mu_std': mu_cat.std().item(),
                        f'{name}_sigma_mean': sigma_cat.mean().item(),
                        f'{name}_sigma_std': sigma_cat.std().item(),
                        f'{name}_nu_mean': nu_cat.mean().item(),
                        f'{name}_nu_std': nu_cat.std().item()
                    }
                elif name == 'NormalModel':
                    mu_cat = torch.cat([batch[0] for batch in output_params[name]])
                    sigma_cat = torch.cat([batch[1] for batch in output_params[name]])
                    output_stats = {
                        f'{name}_mu_mean': mu_cat.mean().item(),
                        f'{name}_mu_std': mu_cat.std().item(),
                        f'{name}_sigma_mean': sigma_cat.mean().item(),
                        f'{name}_sigma_std': sigma_cat.std().item()
                    }
                else:  # NormalModelGlobalSigma
                    mu_cat = torch.cat([batch[0] for batch in output_params[name]])
                    output_stats = {
                        f'{name}_mu_mean': mu_cat.mean().item(),
                        f'{name}_mu_std': mu_cat.std().item(),
                        f'{name}_global_sigma': models[name].sigma.item()
                    }

                wandb.log({
                    'epoch': epoch,
                    'flops': total_flops[name],
                    f'{name}_train_loss': avg_train_loss,
                    f'{name}_test_loss': avg_test_loss,
                    **output_stats
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
        # Create scatter plots
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                if name == 'TModel':
                    mu, _, _ = model(X_test_tensor.to(device))
                elif name == 'NormalModel':
                    mu, _ = model(X_test_tensor.to(device))
                else:  # NormalModelGlobalSigma
                    mu, _ = model(X_test_tensor.to(device))
                
                mu = mu.cpu().numpy()
                y_true = y_test_tensor.cpu().numpy()

                fig, ax = plt.subplots(figsize=(10, 10))
                ax.scatter(y_true, mu, alpha=0.5)
                ax.set_xlabel('True Values')
                ax.set_ylabel('Predicted Mu')
                ax.set_title(f'{name} - Mu Predictions vs True Values')
                
                # Set y-axis limits to match the range of true y values
                y_min, y_max = y_true.min(), y_true.max()
                ax.set_ylim(y_min, y_max)
                
                # Plot the perfect prediction line
                ax.plot([y_min, y_max], [y_min, y_max], 'r--', lw=2)
                
                # Add text showing the range of y values
                ax.text(0.05, 0.95, f'Y range: {y_min:.2f} to {y_max:.2f}', 
                        transform=ax.transAxes, verticalalignment='top')
                
                # Save the plot to a buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                
                # Convert BytesIO to PIL Image
                image = Image.open(buf)
                
                # Log the plot to wandb
                wandb.log({f"{name}_mu_scatter": wandb.Image(image)})
                
                plt.close(fig)
                buf.close()

        wandb.log({
            'best_model': best_model_name,
            'best_test_loss': best_test_loss,
            'total_flops': max(total_flops.values())
        })

    return models, best_model_name, best_test_loss