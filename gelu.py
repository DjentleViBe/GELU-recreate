import torch
import numpy as np
import torch.nn as nn
from file_operations import create_directory
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from NeuralNet import CIFAR10CNN, adjust_lr, prepare_datasets
from csv_operations import csv_write2

def save(model, optimizer, epoch_loss, activation_type, epoch, dir):
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch_loss': epoch_loss
    }, dir + activation_type + '_' + str(epoch) + '.pth')

def cifar10_data(epochs, learn_rate, device, activation_type='default'):
    loss_collect = []
    val_collect = []
    test_collect = []
    dir = 'RESULTS/CIFAR10/' + activation_type + '/'
    create_directory('RESULTS/CIFAR10/' + activation_type + '/')
    create_directory('PICS/CIFAR10/' + activation_type + '/')

    train_dataset, val_dataset, test_dataset, train_indices, val_indices = prepare_datasets(
            val_ratio=0.0,
            mode = 0
        )
    # Step 2: Prepare datasets
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # -------------------------
    # Training setup
    # -------------------------
    model = CIFAR10CNN().to(device)
    optimizer = Adam(model.parameters(), lr=learn_rate)
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Training loop skeleton
    # -------------------------
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        adjust_lr(optimizer, epoch)
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * x.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        loss_collect.append(epoch_loss)
        
        # Optional: validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = outputs.max(1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()
        val_acc = correct_val / max(total_val, 1.0)
        val_collect.append(val_acc)
        if epoch % 5 == 0:
            model.eval()
            correct_test, total_test = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    outputs = model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += target.size(0)
                    correct_test += (predicted == target).sum().item()
        test_collect.append(100 * correct_test / total_test)
        if (epoch + 1) % 20  == 0:
            save(model, optimizer, epoch_loss, activation_type, epoch, dir)    
        print(f"Epoch {epoch+1}, loss: {epoch_loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {100 * correct_test / total_test:.4f}, lr : {lr:.5f}")

    loss_collect = torch.tensor(loss_collect)
    test_collect = torch.tensor(test_collect)
    val_collect = torch.tensor(val_collect)
    csv_write2(dir + '/loss_history_' + activation_type + '.csv', 
              torch.linspace(1, 200, 200), 
              loss_collect, 'epoch', 'loss', '', 'test', val_collect, test_collect)
    