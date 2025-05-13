import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast  # Updated AMP imports
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Physical constants (aligned with Lin et al., 2018)
WAVELENGTH = 750e-6    # 0.4 THz (750 μm)
PIXEL_SIZE = 400e-6    # 400 μm pixel pitch
LAYER_DISTANCE = 10e-3 # 30 mm between layers
LAYER_SIZE = 200       # 200x200 pixels per layer
NUM_LAYERS = 5         # 5 diffractive layers
BATCH_SIZE = 128       # Increased for better GPU utilization
NUM_EPOCHS = 50

# Phase mask module
class PhaseMask(nn.Module):
    def __init__(self, size):
        super(PhaseMask, self).__init__()
        self.phase = nn.Parameter(torch.empty(size, size))
        nn.init.uniform_(self.phase, a=0, b=2 * np.pi)  # 0 to 2π
    
    def forward(self, field):
        return field * torch.exp(1j * self.phase)

# Angular spectrum propagation
class AngularSpectrumPropagation(nn.Module):
    def __init__(self, layer_size, wavelength, pixel_size, distance):
        super(AngularSpectrumPropagation, self).__init__()
        self.layer_size = layer_size
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.distance = distance
        
        # Precompute transfer function
        k = 2 * np.pi / wavelength
        fx = torch.fft.fftfreq(layer_size, d=pixel_size).to(device)
        fy = torch.fft.fftfreq(layer_size, d=pixel_size).to(device)
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY
        kz = torch.sqrt(k**2 - kx**2 - ky**2 + 0j)
        kz = torch.where(kz.imag == 0, kz, 0j)  # Evanescent waves
        self.H = torch.exp(1j * kz * distance).to(device)
    
    def forward(self, field):
        # FFT of input field
        field_fft = torch.fft.fft2(field)
        # Apply transfer function
        field_fft = field_fft * self.H
        # Inverse FFT
        output = torch.fft.ifft2(field_fft)
        # Check for NaNs
        if torch.isnan(output).any():
            raise ValueError("NaN detected in propagation")
        return output

# Diffractive layer
class DiffractiveLayer(nn.Module):
    def __init__(self, layer_size, wavelength, pixel_size, distance):
        super(DiffractiveLayer, self).__init__()
        self.phase_mask = PhaseMask(layer_size)
        self.propagation = AngularSpectrumPropagation(
            layer_size, wavelength, pixel_size, distance
        )
    
    def forward(self, field):
        field = self.phase_mask(field)
        field = self.propagation(field)
        return field

# D²NN model
class D2NN(nn.Module):
    def __init__(self, num_layers, layer_size, wavelength, pixel_size, 
                 layer_distance, num_classes=10):
        super(D2NN, self).__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.num_classes = num_classes
        
        # Diffractive layers
        self.layers = nn.ModuleList([
            DiffractiveLayer(layer_size, wavelength, pixel_size, layer_distance)
            for _ in range(num_layers)
        ])
        
        # Detector regions (2x5 grid, 50x50 pixels each for better separation)
        self.detector_size = layer_size // 4  # 50x50 pixels
        self.detector_regions = []
        for i in range(num_classes):
            row = i // 5
            col = i % 5
            y_start = row * self.detector_size + 25  # Center in 100x100 block
            x_start = col * self.detector_size + 25
            y_end = y_start + self.detector_size
            x_end = x_start + self.detector_size
            self.detector_regions.append((y_start, y_end, x_start, x_end))
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Softer normalization to preserve features
        x = x / (torch.linalg.vector_norm(x, ord=2, dim=(2, 3), 
                                         keepdim=True) + 1e-6)
        
        # Convert to complex field
        field = x.reshape(batch_size, 1, self.layer_size, 
                          self.layer_size).to(device)
        field = field * torch.exp(1j * torch.zeros_like(field))
        
        # Store intermediate fields
        intermediate_fields = [field.clone()]
        
        # Forward pass
        for layer in self.layers:
            field = layer(field)
            intermediate_fields.append(field.clone())
        
        # Calculate intensity
        intensity = torch.abs(field) ** 2
        
        # Calculate energy in each detector region
        outputs = torch.zeros(batch_size, self.num_classes, device=device)
        for i, (y_start, y_end, x_start, x_end) in enumerate(self.detector_regions):
            region_energy = torch.sum(
                intensity[:, 0, y_start:y_end, x_start:x_end],
                dim=(1, 2)
            )
            outputs[:, i] = region_energy
        
        return outputs, intermediate_fields

# Custom loss function
class EnergyMaximizationLoss(nn.Module):
    def __init__(self):
        super(EnergyMaximizationLoss, self).__init__()
    
    def forward(self, outputs, targets):
        # Normalize outputs to probabilities
        outputs = outputs / (torch.sum(outputs, dim=1, keepdim=True) + 1e-6)
        # Cross-entropy on normalized outputs
        log_probs = torch.log(outputs + 1e-6)
        loss = nn.NLLLoss()(log_probs, targets)
        return loss

# Learning rate warmup
def warmup_schedule(epoch):
    if epoch < 5:
        return (epoch + 1) / 5
    return 1.0

# Load datasets
def load_datasets(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Resize((LAYER_SIZE, LAYER_SIZE)),
        transforms.ToTensor(),
    ])
    
    # MNIST
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    mnist_train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False
    )
    
    # Fashion-MNIST
    fashion_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    fashion_test = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    fashion_train_loader = torch.utils.data.DataLoader(
        fashion_train, batch_size=batch_size, shuffle=True
    )
    fashion_test_loader = torch.utils.data.DataLoader(
        fashion_test, batch_size=batch_size, shuffle=False
    )
    
    # Rotated MNIST
    transform_rotated = transforms.Compose([
        transforms.RandomRotation(degrees=(-90, 90)),
        transforms.Resize((LAYER_SIZE, LAYER_SIZE)),
        transforms.ToTensor(),
    ])
    rotated_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform_rotated
    )
    rotated_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform_rotated
    )
    rotated_train_loader = torch.utils.data.DataLoader(
        rotated_train, batch_size=batch_size, shuffle=True
    )
    rotated_test_loader = torch.utils.data.DataLoader(
        rotated_test, batch_size=batch_size, shuffle=False
    )
    
    return {
        'mnist': (mnist_train_loader, mnist_test_loader),
        'fashion_mnist': (fashion_train_loader, fashion_test_loader),
        'rotated_mnist': (rotated_train_loader, rotated_test_loader)
    }

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, 
                scheduler, num_epochs=NUM_EPOCHS, device='cpu', save_path=None):
    scaler = GradScaler('cuda')  # Updated API
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with autocast('cuda'):  # Updated API
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                t.set_postfix(loss=loss.item(), acc=correct/total)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad(), tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]") as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                t.set_postfix(loss=loss.item(), acc=test_correct/test_total)
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_correct / test_total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc and save_path:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
        
        scheduler.step()
    
    return history

# Evaluation function
def evaluate_model(model, test_loader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    return accuracy, cm

# Plot learning curves
def plot_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_title('Loss vs. Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_title('Accuracy vs. Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig

# Visualize fields
def visualize_fields(model, images, labels):
    model.eval()
    with torch.no_grad():
        outputs, fields = model(images.to(device))
    
    num_images = len(images)
    num_layers = len(fields)
    fig, axes = plt.subplots(num_images, num_layers, 
                            figsize=(num_layers * 3, num_images * 3))
    
    for i in range(num_images):
        for j in range(num_layers):
            ax = axes[i, j] if num_images > 1 else axes[j]
            intensity = torch.abs(fields[j][i, 0].cpu()) ** 2
            ax.imshow(intensity, cmap='inferno')
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Layer {j+1}' if j < num_layers-1 else 'Output')
        axes[i, 0].set_ylabel(f'Label: {labels[i]}')
    
    # Log output energies for debugging
    print(f"Sample outputs:\n{outputs.cpu().numpy()[:5]}")
    plt.tight_layout()
    return fig

# Transfer learning experiment
def run_transfer_learning_experiment(base_model_path, num_frozen_layers, 
                                    target_dataset, num_epochs=30, 
                                    learning_rate=0.001, save_path=None):
    base_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, 
                      LAYER_DISTANCE)
    base_model.load_state_dict(torch.load(base_model_path))
    base_model.to(device)
    
    transfer_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, 
                          LAYER_DISTANCE)
    transfer_model.load_state_dict(base_model.state_dict())
    transfer_model.to(device)
    
    # Freeze layers
    for i in range(num_frozen_layers):
        for param in transfer_model.layers[i].parameters():
            param.requires_grad = False
    
    train_loader, test_loader = target_dataset
    criterion = EnergyMaximizationLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, transfer_model.parameters()),
        lr=learning_rate, momentum=0.9
    )
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    
    history = train_model(
        transfer_model, train_loader, test_loader, criterion, optimizer, 
        scheduler, num_epochs=num_epochs, device=device, save_path=save_path
    )
    
    accuracy, cm = evaluate_model(transfer_model, test_loader, device=device)
    return transfer_model, history, accuracy, cm

# Main experiment
def run_full_experiment():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Loading datasets...")
    datasets = load_datasets(batch_size=BATCH_SIZE)
    
    class_names = {
        'mnist': [str(i) for i in range(10)],
        'fashion_mnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'rotated_mnist': [str(i) for i in range(10)]
    }
    
    print("\n=== Training Baseline MNIST Model ===")
    mnist_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, 
                       LAYER_DISTANCE)
    mnist_model.to(device)
    
    criterion = EnergyMaximizationLoss()
    optimizer = optim.SGD(mnist_model.parameters(), lr=0.02, momentum=0.9)  # Increased LR
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    
    mnist_history = train_model(
        mnist_model, datasets['mnist'][0], datasets['mnist'][1], 
        criterion, optimizer, scheduler, num_epochs=NUM_EPOCHS, 
        device=device, save_path='models/mnist_baseline.pth'
    )
    
    fig = plot_learning_curves(mnist_history)
    fig.savefig('results/mnist_learning_curves.png')
    plt.close(fig)
    
    mnist_acc, mnist_cm = evaluate_model(mnist_model, datasets['mnist'][1], 
                                         device=device)
    print(f"MNIST Test Accuracy: {mnist_acc:.4f}")
    
    fig = plot_confusion_matrix(mnist_cm, class_names['mnist'])
    fig.savefig('results/mnist_confusion_matrix.png')
    plt.close(fig)
    
    # Visualize fields
    sample_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            torchvision.datasets.MNIST(
                root='./data', train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize((LAYER_SIZE, LAYER_SIZE)),
                    transforms.ToTensor(),
                ])
            ),
            indices=range(10)
        ),
        batch_size=10, shuffle=False
    )
    
    sample_images, sample_labels = next(iter(sample_loader))
    fig = visualize_fields(mnist_model, sample_images[:5], 
                          [class_names['mnist'][l] for l in sample_labels[:5]])
    fig.savefig('results/mnist_field_visualization.png')
    plt.close(fig)
    
    # Transfer learning experiments
    print("\n=== Transfer Learning: Fashion-MNIST ===")
    fashion_model, fashion_history, fashion_acc, fashion_cm = run_transfer_learning_experiment(
        base_model_path='models/mnist_baseline.pth',
        num_frozen_layers=3,
        target_dataset=datasets['fashion_mnist'],
        num_epochs=30,
        learning_rate=0.001,
        save_path='models/fashion_mnist_transfer.pth'
    )
    
    fig = plot_learning_curves(fashion_history)
    fig.savefig('results/fashion_mnist_learning_curves.png')
    plt.close(fig)
    
    fig = plot_confusion_matrix(fashion_cm, class_names['fashion_mnist'])
    fig.savefig('results/fashion_mnist_confusion_matrix.png')
    plt.close(fig)
    print(f"Fashion-MNIST Test Accuracy: {fashion_acc:.4f}")
    
    print("\n=== Transfer Learning: Rotated-MNIST ===")
    rotated_model, rotated_history, rotated_acc, rotated_cm = run_transfer_learning_experiment(
        base_model_path='models/mnist_baseline.pth',
        num_frozen_layers=3,
        target_dataset=datasets['rotated_mnist'],
        num_epochs=30,
        learning_rate=0.001,
        save_path='models/rotated_mnist_transfer.pth'
    )
    
    fig = plot_learning_curves(rotated_history)
    fig.savefig('results/rotated_mnist_learning_curves.png')
    plt.close(fig)
    
    fig = plot_confusion_matrix(rotated_cm, class_names['rotated_mnist'])
    fig.savefig('results/rotated_mnist_confusion_matrix.png')
    plt.close(fig)
    print(f"Rotated-MNIST Test Accuracy: {rotated_acc:.4f}")

if __name__ == '__main__':
    run_full_experiment()