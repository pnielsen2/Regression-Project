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

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs, max_flops):
    train_losses = []
    test_losses = []
    total_flops = 0
    flops_per_forward = count_flops(model, train_loader.dataset.tensors[0].shape[1])
    
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            if isinstance(model, TModel):
                mu, sigma, nu = model(X_batch)
                loss = criterion(y_batch, mu, sigma, nu)
            else:
                mu, sigma = model(X_batch)
                loss = criterion(y_batch, mu, sigma)

            if torch.isnan(loss):
                print(f'Epoch {epoch+1}: Detected NaN loss, stopping training.')
                return train_losses, test_losses, total_flops

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_flops += flops_per_forward * X_batch.size(0) * 2  # *2 for forward and backward pass
            
            if total_flops > max_flops:
                print(f'Reached max FLOPs ({max_flops}), stopping training.')
                return train_losses, test_losses, total_flops
        
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            test_loss = 0
            for X_batch, y_batch in test_loader:
                if isinstance(model, TModel):
                    mu, sigma, nu = model(X_batch)
                    batch_loss = criterion(y_batch, mu, sigma, nu)
                else:
                    mu, sigma = model(X_batch)
                    batch_loss = criterion(y_batch, mu, sigma)

                if torch.isnan(batch_loss):
                    print(f'Epoch {epoch+1}: Detected NaN test loss, stopping training.')
                    return train_losses, test_losses, total_flops

                test_loss += batch_loss.item()
                total_flops += flops_per_forward * X_batch.size(0)
                
                if total_flops > max_flops:
                    print(f'Reached max FLOPs ({max_flops}), stopping training.')
                    return train_losses, test_losses, total_flops
            
            test_losses.append(test_loss / len(test_loader))
        
        if (epoch + 1) % 10 == 0:
            print(f'{model.__class__.__name__} - Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Test Loss: {test_loss / len(test_loader)}, Total FLOPs: {total_flops}')

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
        for epoch in range(len(train_losses)):
            wandb.log({
                f'{name}_train_loss': train_losses[epoch],
                f'{name}_test_loss': test_losses[epoch],
                f'{name}_total_flops': total_flops,
                f'{name}_flops_per_epoch': total_flops / (epoch + 1)
            })
        if test_losses[-1] < best_test_loss:
            best_test_loss = test_losses[-1]
            best_model_name = name

    wandb.log({
        'best_model': best_model_name,
        'best_test_loss': best_test_loss,
        'total_flops': sum(wandb.run.summary[f'{name}_total_flops'] for name in models.keys())
    })