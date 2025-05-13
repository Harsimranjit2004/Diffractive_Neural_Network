import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
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
LAYER_DISTANCE = 30e-3 # 10 mm between layers
LAYER_SIZE = 200       # 200x200 pixels
NUM_LAYERS = 5        # 2 layers for simplicity
BATCH_SIZE = 128
NUM_EPOCHS = 50

# Phase mask module
class PhaseMask(nn.Module):
    def __init__(self, size):
        super(PhaseMask, self).__init__()
        self.phase = nn.Parameter(torch.empty(size, size))
        nn.init.uniform_(self.phase, a=0, b=2 * np.pi)
    
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
        kz_squared = k**2 - kx**2 - ky**2
        kz_squared = torch.clamp(kz_squared, min=0)
        kz = torch.sqrt(kz_squared + 0j)
        kz = torch.where(kz.imag == 0, kz, 0j)
        self.H = torch.exp(1j * kz * distance).to(device)
        print(f"AngularSpectrumPropagation: H shape={self.H.shape}, "
              f"evanescent waves={(kz.imag != 0).sum().item()}")

    def forward(self, field):
        field_fft = torch.fft.fft2(field)
        field_fft = field_fft * self.H
        output = torch.fft.ifft2(field_fft)
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
        
        self.layers = nn.ModuleList([
            DiffractiveLayer(layer_size, wavelength, pixel_size, layer_distance)
            for _ in range(num_layers)
        ])
        
        # Detector regions (2x5 grid, 60x60 pixels)
        self.detector_size = 60
        self.detector_regions = []
        for i in range(num_classes):
            row = i // 5
            col = i % 5
            y_start = row * 80 + 10
            x_start = col * 80 + 10
            y_end = y_start + self.detector_size
            x_end = x_start + self.detector_size
            self.detector_regions.append((y_start, y_end, x_start, x_end))
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Normalize input
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
    def __init__(self, margin=0.5):
        super(EnergyMaximizationLoss, self).__init__()
        self.margin = margin
    
    def forward(self, outputs, targets):
        batch_size = outputs.shape[0]
        outputs = outputs / (torch.sum(outputs, dim=1, keepdim=True) + 1e-6)
        target_energy = outputs[torch.arange(batch_size), targets]
        mask = torch.ones_like(outputs).scatter_(1, targets.unsqueeze(1), 0)
        non_target_energy = (outputs * mask).max(dim=1)[0]
        loss = torch.mean(torch.relu(self.margin - target_energy + non_target_energy))
        return loss

# Load datasets
def load_datasets(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.Pad(padding=(86, 86, 86, 86)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    mnist_train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    mnist_train_loader = DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    mnist_test_loader = DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )
    
    return mnist_train, mnist_test, mnist_train_loader, mnist_test_loader

# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, 
                scheduler, warmup_scheduler, num_epochs=NUM_EPOCHS, device='cpu', save_path=None):
    scaler = GradScaler('cuda')
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    patience = 10
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Log target energy
                normalized_outputs = outputs / (torch.sum(outputs, dim=1, keepdim=True) + 1e-6)
                mean_target_energy = normalized_outputs[torch.arange(len(labels)), labels].mean().item()
                
                t.set_postfix(loss=loss.item(), acc=correct/total, energy=mean_target_energy)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad(), tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]") as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                t.set_postfix(loss=loss.item(), acc=test_correct/test_total)
        
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_correct / test_total
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc and save_path:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        if epoch < 5:
            warmup_scheduler.step()
        else:
            scheduler.step()
        
        # Visualize every 5 epochs
        if epoch % 5 == 0:
            sample_loader = DataLoader(
                torch.utils.data.Subset(test_loader.dataset, range(10)),
                batch_size=10, shuffle=False
            )
            sample_images, sample_labels = next(iter(sample_loader))
            fig, fig2 = visualize_fields(model, sample_images, sample_labels, 
                                        [str(i) for i in range(10)], epoch)
            fig.savefig(f'results/fields_epoch_{epoch}.png')
            fig2.savefig(f'results/energies_epoch_{epoch}.png')
            plt.close(fig)
            plt.close(fig2)
            
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            cm_fig = plot_confusion_matrix(all_labels, all_preds, [str(i) for i in range(10)], epoch)
            cm_fig.savefig(f'results/cm_epoch_{epoch}.png')
            plt.close(cm_fig)
    
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
def plot_confusion_matrix(cm, class_names, epoch):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Epoch {epoch} - Confusion Matrix')
    plt.tight_layout()
    return fig

# Visualize fields
def visualize_fields(model, images, labels, class_names, epoch):
    model.eval()
    with torch.no_grad():
        outputs, fields = model(images.to(device))
    
    num_images = min(len(images), 5)
    num_layers = len(fields)
    fig, axes = plt.subplots(num_images, num_layers, 
                            figsize=(num_layers * 3, num_images * 3))
    
    for i in range(num_images):
        for j in range(num_layers):
            ax = axes[i, j] if num_images > 1 else axes[j]
            intensity = torch.abs(fields[j][i, 0].cpu()) ** 2
            ax.imshow(intensity, cmap='inferno')
            # Plot detector regions on output layer
            if j == num_layers - 1:
                for k, (y_start, y_end, x_start, x_end) in enumerate(model.detector_regions):
                    rect = plt.Rectangle((x_start, y_start), 
                                        x_end-x_start, y_end-y_start,
                                        edgecolor='blue', facecolor='none')
                    ax.add_patch(rect)
            ax.axis('off')
            if i == 0:
                ax.set_title(f'Layer {j+1}' if j < num_layers-1 else 'Output')
        axes[i, 0].set_ylabel(f'Label: {class_names[labels[i]]}')
    
    normalized_outputs = outputs / (torch.sum(outputs, dim=1, keepdim=True) + 1e-6)
    print(f"Epoch {epoch} - Sample outputs (normalized):\n{normalized_outputs.cpu().numpy()[:5]}")
    print(f"Epoch {epoch} - Predicted classes: {torch.argmax(outputs, dim=1)[:5].cpu().numpy()}")
    print(f"Epoch {epoch} - True labels: {labels[:5].numpy()}")
    
    # Plot detector energies
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for i in range(min(5, len(outputs))):
        ax2.plot(normalized_outputs[i].cpu().numpy(), label=f'Sample {i} (Label {labels[i]})')
    ax2.set_xlabel('Detector')
    ax2.set_ylabel('Normalized Energy')
    ax2.legend()
    ax2.set_title(f'Epoch {epoch} - Detector Energies')
    
    plt.tight_layout()
    return fig, fig2

# Main experiment
def run_experiment():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Loading MNIST dataset...")
    mnist_train, mnist_test, train_loader, test_loader = load_datasets(batch_size=BATCH_SIZE)
    
    print("\n=== Training MNIST Model ===")
    model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, 
                 LAYER_DISTANCE)
    model.to(device)
    
    criterion = EnergyMaximizationLoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS-5)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5*len(train_loader))
    
    history = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, 
        warmup_scheduler, num_epochs=NUM_EPOCHS, device=device, 
        save_path='models/mnist_best.pth'
    )
    
    fig = plot_learning_curves(history)
    fig.savefig('results/mnist_learning_curves.png')
    plt.close(fig)
    
    acc, cm = evaluate_model(model, test_loader, device=device)
    print(f"MNIST Test Accuracy: {acc:.4f}")
    
    fig = plot_confusion_matrix(cm, [str(i) for i in range(10)], NUM_EPOCHS)
    fig.savefig('results/mnist_confusion_matrix.png')
    plt.close(fig)
    
    # Visualize fields
    sample_loader = DataLoader(
        torch.utils.data.Subset(mnist_test, range(10)),
        batch_size=10, shuffle=False
    )
    sample_images, sample_labels = next(iter(sample_loader))
    fig, fig2 = visualize_fields(model, sample_images, sample_labels, 
                                [str(i) for i in range(10)], NUM_EPOCHS)
    fig.savefig('results/mnist_field_visualization.png')
    fig2.savefig('results/mnist_energies.png')
    plt.close(fig)
    plt.close(fig2)

if __name__ == '__main__':
    run_experiment()