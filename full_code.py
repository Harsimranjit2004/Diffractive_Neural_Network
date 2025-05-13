"""
Diffractive Deep Neural Networks (D²NNs) with Transfer Learning Capabilities
- Implementation of the experiment described in the paper
- Uses PyTorch for efficient GPU training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from time import time
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
torch.set_default_dtype(torch.float64)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

from torch.optim.lr_scheduler import LambdaLR

def warmup_schedule(epoch):
    warmup_epochs = 5
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0



# Check for GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants for the D²NN
WAVELENGTH = 532e-9  # 532 nm green light
PIXEL_SIZE = 40e-6   # 40 μm pixel pitch
LAYER_DISTANCE = 40e-3  # 40 mm between layers
LAYER_SIZE = 200     # 200x200 pixels per layer
NUM_LAYERS = 5       # 5 diffractive layers


class CustomDataset(Dataset):
    """Custom dataset wrapper for MNIST, Fashion-MNIST, and Rotated-MNIST"""
    
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class EnergyMaximizationLoss(nn.Module):
    def __init__(self):
        super(EnergyMaximizationLoss, self).__init__()

    def forward(self, outputs, targets):
        batch_size = outputs.size(0)

        # Normalize outputs to sum to 1 across classes
        outputs = outputs / (outputs.sum(dim=1, keepdim=True) + 1e-8)

        # Gather energy at the correct detector for each sample
        correct_energy = torch.clamp(outputs[torch.arange(batch_size), targets], min=1e-6)


        # Loss is negative log of correct energy (to maximize it)
        loss = -torch.log(correct_energy + 1e-8).mean()
        return loss

    

def rotate_mnist(mnist_dataset, min_angle=-45, max_angle=45):
    """Create rotated MNIST dataset with random rotations"""
    rotated_data = []
    for img in mnist_dataset.data:
        angle = np.random.uniform(min_angle, max_angle)
        rotated_img = transforms.functional.rotate(img.unsqueeze(0), angle).squeeze(0)
        rotated_data.append(rotated_img)
    
    rotated_data = torch.stack(rotated_data)
    return CustomDataset(rotated_data, mnist_dataset.targets)


def load_datasets(batch_size=64):
    """Load MNIST, Fashion-MNIST, and create Rotated-MNIST datasets"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((LAYER_SIZE, LAYER_SIZE)),
        transforms.ToTensor(),
    ])
    
    # Load MNIST
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Load Fashion-MNIST
    fashion_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create Rotated-MNIST
    rotated_mnist_train = rotate_mnist(torchvision.datasets.MNIST(root='./data', train=True, download=True))
    rotated_mnist_test = rotate_mnist(torchvision.datasets.MNIST(root='./data', train=False, download=True))
    
    # Apply transforms to rotated MNIST
    rotated_transform = transforms.Compose([
        transforms.Resize((LAYER_SIZE, LAYER_SIZE)),
        transforms.ToTensor(),
    ])
    
    rotated_mnist_train = CustomDataset(
        rotated_mnist_train.data, 
        rotated_mnist_train.targets,
        transform=rotated_transform
    )
    
    rotated_mnist_test = CustomDataset(
        rotated_mnist_test.data, 
        rotated_mnist_test.targets,
        transform=rotated_transform
    )
    
    # Create DataLoaders
    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    
    fashion_train_loader = DataLoader(fashion_train, batch_size=batch_size, shuffle=True)
    fashion_test_loader = DataLoader(fashion_test, batch_size=batch_size, shuffle=False)
    
    rotated_train_loader = DataLoader(rotated_mnist_train, batch_size=batch_size, shuffle=True)
    rotated_test_loader = DataLoader(rotated_mnist_test, batch_size=batch_size, shuffle=False)
    
    return {
        'mnist': (mnist_train_loader, mnist_test_loader),
        'fashion_mnist': (fashion_train_loader, fashion_test_loader),
        'rotated_mnist': (rotated_train_loader, rotated_test_loader)
    }


class AngularSpectrumPropagation(nn.Module):
    """PyTorch module implementing angular spectrum propagation"""
    
    def __init__(self, input_size, wavelength, pixel_size, propagation_distance):
        super(AngularSpectrumPropagation, self).__init__()
        self.input_size = input_size
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.propagation_distance = propagation_distance
        
        # Precompute transfer function
        self.transfer_function = self._create_transfer_function().to(device)
        
    def _create_transfer_function(self):
        # Wave number
        k = 2 * np.pi / self.wavelength
        
        # Spatial frequencies
        fx = torch.fft.fftfreq(self.input_size, self.pixel_size)
        fy = torch.fft.fftfreq(self.input_size, self.pixel_size)
        fx_grid, fy_grid = torch.meshgrid(fx, fy, indexing='ij')
        
        # Compute transfer function
        fx_grid = fx_grid.to(device)
        fy_grid = fy_grid.to(device)
        
        # Compute kz component for propagation
        kz_sq = k**2 - (2*np.pi*fx_grid)**2 - (2*np.pi*fy_grid)**2
        kz = torch.sqrt(torch.clamp(kz_sq, min=0.0))
        
        # Transfer function for propagation
        transfer_function = torch.exp(1j * kz * self.propagation_distance)
        
        # Set evanescent waves to zero
        transfer_function[kz_sq <= 0] = 0
        
        return transfer_function
    
    def forward(self, field):
        # Apply FFT
        field_fft = torch.fft.fft2(field)
        
        # Apply transfer function
        field_fft_propagated = field_fft * self.transfer_function
        
        # Apply inverse FFT
        field_propagated = torch.fft.ifft2(field_fft_propagated)
        
        return field_propagated


class PhaseMask(nn.Module):
    """Learnable phase mask module"""
    
    def __init__(self, size):
        super(PhaseMask, self).__init__()
        # Initialize phase values between 0 and 2π
        # self.phase = nn.Parameter(torch.rand(size, size) * 2 * np.pi)
        self.phase = nn.Parameter(torch.empty(size, size))
        nn.init.uniform_(self.phase, a=-0.1, b=0.1)


        
    def forward(self, field):
        # Apply phase modulation: field * exp(j*phi)
        return field * torch.exp(1j * self.phase)


class DiffractiveLayer(nn.Module):
    """Complete diffractive layer (phase mask + propagation)"""
    
    def __init__(self, size, wavelength, pixel_size, propagation_distance):
        super(DiffractiveLayer, self).__init__()
        self.phase_mask = PhaseMask(size)
        self.propagation = AngularSpectrumPropagation(
            size, wavelength, pixel_size, propagation_distance
        )
        
    def forward(self, field):
        # Apply phase mask
        field = self.phase_mask(field)
        # Propagate to next layer
        field = self.propagation(field)
        return field


class D2NN(nn.Module):
    """Complete Diffractive Deep Neural Network"""
    # def __init__(self, num_layers, layer_size, wavelength, pixel_size, layer_distance, num_classes=10):
    #     super(D2NN, self).__init__()
    #     self.num_layers = num_layers
    #     self.layer_size = layer_size
    #     self.num_classes = num_classes

    #     # Create diffractive layers
    #     self.layers = nn.ModuleList([
    #         DiffractiveLayer(layer_size, wavelength, pixel_size, layer_distance)
    #         for _ in range(num_layers)
    #     ])

    #     self.detector_size = layer_size // 4
    #     regions_per_row = int(np.ceil(np.sqrt(num_classes)))
    #     self.detector_regions = []

    #     for i in range(num_classes):
    #         row = i // regions_per_row
    #         col = i % regions_per_row
    #         y_start = row * self.detector_size
    #         x_start = col * self.detector_size
    #         y_end = min((row + 1) * self.detector_size, layer_size)
    #         x_end = min((col + 1) * self.detector_size, layer_size)
    #         self.detector_regions.append((y_start, y_end, x_start, x_end))

    # def forward(self, x):
    #     batch_size = x.shape[0]

    #     # Normalize input
    #     x = x / (torch.linalg.vector_norm(x, ord=2, dim=(1, 2, 3), keepdim=True) + 1e-8)

    #     field = x.reshape(batch_size, 1, self.layer_size, self.layer_size).to(x.device)
    #     field = field * torch.exp(1j * torch.zeros_like(field))

    #     intermediate_fields = [field]
    #     for layer in self.layers:
    #         field = layer(field)
    #         intermediate_fields.append(field)

    #     intensity = torch.abs(field) ** 2
    #     outputs = torch.zeros(batch_size, self.num_classes, device=x.device)
    #     for i, (y_start, y_end, x_start, x_end) in enumerate(self.detector_regions):
    #         region_energy = torch.sum(intensity[:, 0, y_start:y_end, x_start:x_end], dim=(1, 2))
    #         outputs[:, i] = region_energy

    #     return outputs, intermediate_fields

    # def get_intermediate_fields(self, x):
    #     with torch.no_grad():
    #         _, fields = self.forward(x)
    #     return fields 
    def __init__(self, num_layers, layer_size, wavelength, pixel_size, layer_distance, num_classes=10):
        super(D2NN, self).__init__()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.num_classes = num_classes
        
        # Create diffractive layers
        self.layers = nn.ModuleList([
            DiffractiveLayer(layer_size, wavelength, pixel_size, layer_distance)
            for _ in range(num_layers)
        ])
        
        # Define detector regions (1/10 of the plane for each class)
        # self.detector_size = layer_size // 5  # Make regions smaller
        self.detector_size = layer_size // num_classes  # = 20 for 200x200

        regions_per_row = num_classes  # Spread them in 1 row

        self.detector_regions = []
        for i in range(num_classes):
            y_start = 0
            x_start = i * self.detector_size
            y_end = y_start + self.detector_size
            x_end = x_start + self.detector_size
            self.detector_regions.append((y_start, y_end, x_start, x_end))

    
    def forward(self, x):
        x = x / (torch.linalg.vector_norm(x, ord=2, dim=(1, 2, 3), keepdim=True) + 1e-8)

        batch_size = x.shape[0]
        
        # Convert input to complex field with uniform phase
        field = x.reshape(batch_size, 1, self.layer_size, self.layer_size).to(device)
        field = field * torch.exp(1j * torch.zeros_like(field))
        
        # Propagate through layers
        intermediate_fields = [field]  # Store for visualization
        for layer in self.layers:
            field = layer(field)
            intermediate_fields.append(field)
            if batch_size == 1:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6, 5))
                plt.imshow(torch.abs(field[0, 0].detach().cpu())**2, cmap='inferno')
                plt.title("Final Intensity at Detector Plane")
                plt.colorbar()
                plt.show()

        
        # Calculate intensity at detector plane
        intensity = torch.abs(field) ** 2
        import torch.nn.functional as F
        intensity = F.avg_pool2d(intensity, kernel_size=3, stride=1, padding=1)

        
        # Calculate energy in each detector region
        outputs = torch.zeros(batch_size, self.num_classes, device=device)
        for i, (y_start, y_end, x_start, x_end) in enumerate(self.detector_regions):
            region_energy = torch.sum(
                intensity[:, 0, y_start:y_end, x_start:x_end], 
                dim=(1, 2)
            )
            outputs[:, i] = region_energy
        
        # Normalize outputs so they sum to 1
        # outputs = outputs / (torch.sum(outputs, dim=1, keepdim=True) + 1e-8)
       # Normalize outputs so they sum to 1
        outputs = outputs / (torch.sum(outputs, dim=1, keepdim=True) + 1e-8)
 
        return outputs, intermediate_fields

    def get_intermediate_fields(self, x):
        """Get all intermediate fields for visualization"""
        with torch.no_grad():
            _, fields = self.forward(x)
        return fields


def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, 
                num_epochs=50, device='cpu', save_path=None):
    """Train the D²NN model"""
    
    # Keep track of training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm for progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as t:
            for inputs, labels in t:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = model(inputs)
                loss = criterion(torch.log(outputs + 1e-8), labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                t.set_postfix(loss=loss.item(), acc=correct/total)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        # Testing phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            with tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]") as t:
                for inputs, labels in t:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Forward pass
                    outputs, _ = model(inputs)
                    loss = criterion(torch.log(outputs + 1e-8), labels)
                    
                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    t.set_postfix(loss=loss.item(), acc=correct/total)
        
        epoch_loss = running_loss / len(test_loader.dataset)
        epoch_acc = correct / total
        history['test_loss'].append(epoch_loss)
        history['test_acc'].append(epoch_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Save best model
        if epoch_acc > best_acc and save_path:
            best_acc = epoch_acc
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with accuracy: {best_acc:.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {history['train_loss'][-1]:.4f}, "
              f"Train Acc: {history['train_acc'][-1]:.4f}, "
              f"Test Loss: {history['test_loss'][-1]:.4f}, "
              f"Test Acc: {history['test_acc'][-1]:.4f}")
    
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate the D²NN model"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, cm


def visualize_fields(model, sample_images, class_names=None):
    """Visualize the intermediate optical fields"""
    model.eval()
    with torch.no_grad():
        sample_images = sample_images.to(device)
        fields = model.get_intermediate_fields(sample_images)
    
    num_samples = sample_images.shape[0]
    num_layers = len(fields)
    
    fig, axes = plt.subplots(num_samples, num_layers, figsize=(num_layers*3, num_samples*3))
    
    for i in range(num_samples):
        for j in range(num_layers):
            field = fields[j][i, 0].cpu().numpy()
            amplitude = np.abs(field)
            phase = np.angle(field)
            
            # Normalize amplitude for visualization
            amplitude = amplitude / amplitude.max()
            
            # Create HSV image (hue for phase, value for amplitude)
            hsv = np.zeros((field.shape[0], field.shape[1], 3))
            hsv[:, :, 0] = (phase + np.pi) / (2 * np.pi)  # Hue is phase
            hsv[:, :, 1] = 1.0  # Saturation is 1
            hsv[:, :, 2] = amplitude  # Value is amplitude
            
            # Convert to RGB
            rgb = plt.cm.hsv(hsv[:, :, 0])
            rgb[:, :, 0:3] *= hsv[:, :, 2][:, :, np.newaxis]
            
            if num_samples == 1:
                if num_layers == 1:
                    ax = axes
                else:
                    ax = axes[j]
            else:
                if num_layers == 1:
                    ax = axes[i]
                else:
                    ax = axes[i, j]
            
            ax.imshow(rgb)
            ax.axis('off')
            
            # Add titles
            if i == 0:
                if j == 0:
                    ax.set_title("Input")
                else:
                    ax.set_title(f"Layer {j}")
            
            if j == 0 and class_names is not None:
                ax.set_ylabel(class_names[i])
    
    plt.tight_layout()
    return fig


def plot_learning_curves(history):
    """Plot the learning curves from training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm, class_names):
    """Plot a confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig


def run_transfer_learning_experiment(base_model_path, num_frozen_layers, target_dataset, 
                                    num_epochs=30, learning_rate=0.01, save_path=None):
    """Run transfer learning experiment with frozen layers"""
    # Load base model
    base_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, LAYER_DISTANCE)
    base_model.load_state_dict(torch.load(base_model_path))
    base_model.to(device)
    
    # Create new model for transfer learning
    transfer_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, LAYER_DISTANCE)
    transfer_model.to(device)
    
    # Copy parameters from base model
    transfer_model.load_state_dict(base_model.state_dict())
    
    # Freeze specified layers
    for i in range(num_frozen_layers):
        for param in transfer_model.layers[i].parameters():
            param.requires_grad = False
    
    # Prepare dataloaders
    train_loader, test_loader = target_dataset
    
    # Set up training
    # criterion = nn.NLLLoss()
    criterion = EnergyMaximizationLoss() 
    # loss = criterion(outputs, labels)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, transfer_model.parameters()),
        lr=learning_rate, momentum=0.9
    )
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    
    # Train model
    history = train_model(
        transfer_model, train_loader, test_loader, criterion, optimizer, scheduler,
        num_epochs=num_epochs, device=device, save_path=save_path
    )
    
    # Evaluate model
    accuracy, cm = evaluate_model(transfer_model, test_loader, device=device)
    
    return transfer_model, history, accuracy, cm


def run_full_experiment():
    """Run the complete transfer learning experiment"""
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    print("Loading datasets...")
    datasets = load_datasets(batch_size=64)
    
    # Define class names for each dataset
    class_names = {
        'mnist': [str(i) for i in range(10)],
        'fashion_mnist': ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'],
        'rotated_mnist': [str(i) for i in range(10)]
    }
    
    # 1. Train baseline MNIST model
    print("\n=== Training Baseline MNIST Model ===")
    mnist_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, LAYER_DISTANCE)
    mnist_model.to(device)
    

    criterion = EnergyMaximizationLoss()


    optimizer = optim.SGD(mnist_model.parameters(), lr=0.01, momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    
    mnist_history = train_model(
        mnist_model, datasets['mnist'][0], datasets['mnist'][1], 
        criterion, optimizer, scheduler, num_epochs=50, 
        device=device, save_path='models/mnist_baseline.pth'
    )
    
    # Plot learning curves
    fig = plot_learning_curves(mnist_history)
    fig.savefig('results/mnist_learning_curves.png')
    
    # Evaluate baseline model
    mnist_acc, mnist_cm = evaluate_model(mnist_model, datasets['mnist'][1], device=device)
    
    # Plot confusion matrix
    fig = plot_confusion_matrix(mnist_cm, class_names['mnist'])
    fig.savefig('results/mnist_confusion_matrix.png')
    
    # Visualize fields for a few samples
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
    
    # 2. Transfer learning experiments
    target_datasets = ['fashion_mnist', 'rotated_mnist']
    frozen_layers_options = [1, 2, 3, 4]
    
    results = {dataset: [] for dataset in target_datasets}
    
    for dataset_name in target_datasets:
        print(f"\n=== Transfer Learning for {dataset_name} ===")
        
        # First, train from scratch as baseline
        print(f"Training {dataset_name} model from scratch...")
        from_scratch_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, LAYER_DISTANCE)
        from_scratch_model.to(device)
        
        optimizer = optim.SGD(from_scratch_model.parameters(), lr=0.01, momentum=0.9)
        scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
        
        from_scratch_history = train_model(
            from_scratch_model, datasets[dataset_name][0], datasets[dataset_name][1], 
            criterion, optimizer, scheduler, num_epochs=50, 
            device=device, save_path=f'models/{dataset_name}_from_scratch.pth'
        )
        
        from_scratch_acc, from_scratch_cm = evaluate_model(
            from_scratch_model, datasets[dataset_name][1], device=device
        )
        
        results[dataset_name].append({
            'name': 'From Scratch',
            'frozen_layers': 0,
            'accuracy': from_scratch_acc,
            'history': from_scratch_history
        })
        
        # Plot learning curves
        fig = plot_learning_curves(from_scratch_history)
        fig.savefig(f'results/{dataset_name}_from_scratch_learning_curves.png')
        
        # Run transfer learning experiments
        for num_frozen in frozen_layers_options:
            print(f"Transfer learning with {num_frozen} frozen layers...")
            
            transfer_model, transfer_history, transfer_acc, transfer_cm = run_transfer_learning_experiment(
                'models/mnist_baseline.pth', num_frozen, datasets[dataset_name],
                num_epochs=30, learning_rate=0.01,
                save_path=f'models/{dataset_name}_transfer_{num_frozen}layers.pth'
            )
            
            results[dataset_name].append({
                'name': f'Transfer (freeze {num_frozen} layers)',
                'frozen_layers': num_frozen,
                'accuracy': transfer_acc,
                'history': transfer_history
            })
            
            # Plot learning curves
            fig = plot_learning_curves(transfer_history)
            fig.savefig(f'results/{dataset_name}_transfer_{num_frozen}layers_learning_curves.png')
            
            # Plot confusion matrix
            fig = plot_confusion_matrix(transfer_cm, class_names[dataset_name])
            fig.savefig(f'results/{dataset_name}_transfer_{num_frozen}layers_confusion_matrix.png')
    
    # 3. Summarize results
    print("\n=== Summary of Results ===")
    for dataset_name in target_datasets:
        print(f"\nResults for {dataset_name}:")
        print("-" * 50)
        print(f"{'Configuration':<30} {'Accuracy':<10}")
        print("-" * 50)
        
        for result in results[dataset_name]:
            print(f"{result['name']:<30} {result['accuracy']:.4f}")
    
    # 4. Compare optical fields across tasks
    print("\n=== Comparing Optical Fields Across Tasks ===")
    
    # Load best transfer models based on results
    best_fashion_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, LAYER_DISTANCE)
    best_fashion_model.load_state_dict(torch.load('models/fashion_mnist_transfer_2layers.pth'))
    best_fashion_model.to(device)
    
    best_rotated_model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, LAYER_DISTANCE)
    best_rotated_model.load_state_dict(torch.load('models/rotated_mnist_transfer_1layers.pth'))
    best_rotated_model.to(device)
    
    # Get sample images from each dataset
    mnist_sample, _ = next(iter(datasets['mnist'][1]))
    fashion_sample, _ = next(iter(datasets['fashion_mnist'][1]))
    rotated_sample, _ = next(iter(datasets['rotated_mnist'][1]))
    
    # Visualize fields for each model and dataset
    models = {
        'MNIST': mnist_model,
        'Fashion-MNIST': best_fashion_model,
        'Rotated-MNIST': best_rotated_model
    }
    
    samples = {
        'MNIST': mnist_sample[:3],
        'Fashion-MNIST': fashion_sample[:3],
        'Rotated-MNIST': rotated_sample[:3]
    }
    
    # Compare field correlations
    print("\nCalculating optical feature correlations between tasks...")
    
    # Function to compute correlation between optical fields
    def compute_field_correlation(field1, field2):
        """Compute correlation between amplitude patterns of optical fields"""
        amp1 = torch.abs(field1).cpu().numpy().flatten()
        amp2 = torch.abs(field2).cpu().numpy().flatten()
        corr = np.corrcoef(amp1, amp2)[0, 1]
        return corr
    
    # Sample images for correlation analysis
    mnist_corr_img, _ = next(iter(datasets['mnist'][1]))
    fashion_corr_img, _ = next(iter(datasets['fashion_mnist'][1]))
    rotated_corr_img, _ = next(iter(datasets['rotated_mnist'][1]))
    
    mnist_corr_img = mnist_corr_img[:100].to(device)
    fashion_corr_img = fashion_corr_img[:100].to(device)
    rotated_corr_img = rotated_corr_img[:100].to(device)
    
    # Get intermediate fields for each model and dataset
    with torch.no_grad():
        _, mnist_fields_on_mnist = mnist_model.get_intermediate_fields(mnist_corr_img)
        _, fashion_fields_on_fashion = best_fashion_model.get_intermediate_fields(fashion_corr_img)
        _, rotated_fields_on_rotated = best_rotated_model.get_intermediate_fields(rotated_corr_img)
        
        _, mnist_fields_on_fashion = mnist_model.get_intermediate_fields(fashion_corr_img)
        _, fashion_fields_on_mnist = best_fashion_model.get_intermediate_fields(mnist_corr_img)
        
        _, mnist_fields_on_rotated = mnist_model.get_intermediate_fields(rotated_corr_img)
        _, rotated_fields_on_mnist = best_rotated_model.get_intermediate_fields(mnist_corr_img)
    
    # Compute correlations for each layer
    num_layers = len(mnist_fields_on_mnist)
    correlations = np.zeros((3, 3, num_layers))
    
    for layer in range(num_layers):
        # MNIST vs Fashion-MNIST
        correlations[0, 1, layer] = compute_field_correlation(
            mnist_fields_on_mnist[layer], fashion_fields_on_mnist[layer]
        )
        
        # MNIST vs Rotated-MNIST
        correlations[0, 2, layer] = compute_field_correlation(
            mnist_fields_on_mnist[layer], rotated_fields_on_mnist[layer]
        )
        
        # Fashion-MNIST vs Rotated-MNIST
        correlations[1, 2, layer] = compute_field_correlation(
            fashion_fields_on_fashion[layer], rotated_fields_on_rotated[layer]
        )
    
    # Make the correlation matrix symmetric
    for i in range(3):
        for j in range(i+1, 3):
            correlations[j, i, :] = correlations[i, j, :]
    
    # Set diagonal to 1
    for i in range(3):
        correlations[i, i, :] = 1.0
    
    # Plot correlation heatmaps for each layer
    task_names = ['MNIST', 'Fashion-MNIST', 'Rotated-MNIST']
    
    plt.figure(figsize=(15, 5 * num_layers))
    for layer in range(num_layers):
        plt.subplot(num_layers, 1, layer + 1)
        sns.heatmap(correlations[:, :, layer], annot=True, fmt='.3f', cmap='viridis',
                   xticklabels=task_names, yticklabels=task_names)
        plt.title(f'Layer {layer} Feature Correlation')
    
    plt.tight_layout()
    plt.savefig('results/task_correlation_heatmaps.png')
    
    # 5. Transfer learning feature analysis
    print("\n=== Analyzing Feature Adaptation in Transfer Learning ===")
    
    # Load models with different numbers of frozen layers
    transfer_models = {}
    for num_frozen in frozen_layers_options:
        model = D2NN(NUM_LAYERS, LAYER_SIZE, WAVELENGTH, PIXEL_SIZE, LAYER_DISTANCE)
        model.load_state_dict(torch.load(f'models/fashion_mnist_transfer_{num_frozen}layers.pth'))
        model.to(device)
        transfer_models[num_frozen] = model
    
    # Visualize phase masks for each layer and model
    plt.figure(figsize=(15, 10))
    
    # Original MNIST model
    for i in range(NUM_LAYERS):
        plt.subplot(NUM_LAYERS, len(frozen_layers_options) + 2, i * (len(frozen_layers_options) + 2) + 1)
        phase = mnist_model.layers[i].phase_mask.phase.detach().cpu().numpy()
        plt.imshow(phase, cmap='hsv', vmin=0, vmax=2*np.pi)
        plt.colorbar()
        if i == 0:
            plt.title('MNIST (Base)')
        plt.xlabel(f'Layer {i+1}')
    
    # Fashion-MNIST from scratch
    for i in range(NUM_LAYERS):
        plt.subplot(NUM_LAYERS, len(frozen_layers_options) + 2, i * (len(frozen_layers_options) + 2) + 2)
        phase = from_scratch_model.layers[i].phase_mask.phase.detach().cpu().numpy()
        plt.imshow(phase, cmap='hsv', vmin=0, vmax=2*np.pi)
        plt.colorbar()
        if i == 0:
            plt.title('Fashion-MNIST (from scratch)')
        plt.xlabel(f'Layer {i+1}')
    
    # Transfer models with different frozen layers
    for j, num_frozen in enumerate(frozen_layers_options):
        for i in range(NUM_LAYERS):
            plt.subplot(NUM_LAYERS, len(frozen_layers_options) + 2, 
                       i * (len(frozen_layers_options) + 2) + j + 3)
            
            phase = transfer_models[num_frozen].layers[i].phase_mask.phase.detach().cpu().numpy()
            plt.imshow(phase, cmap='hsv', vmin=0, vmax=2*np.pi)
            plt.colorbar()
            
            if i == 0:
                plt.title(f'Transfer (freeze {num_frozen})')
            
            # Add indicator for frozen layers
            if i < num_frozen:
                plt.xlabel(f'Layer {i+1} (frozen)')
            else:
                plt.xlabel(f'Layer {i+1}')
    
    plt.tight_layout()
    plt.savefig('results/phase_mask_comparison.png')
    
    # Compute phase mask differences
    print("\nComputing phase mask differences...")
    
    # Compare with original MNIST model
    phase_diffs = {}
    for num_frozen in frozen_layers_options:
        phase_diffs[num_frozen] = []
        for i in range(NUM_LAYERS):
            mnist_phase = mnist_model.layers[i].phase_mask.phase.detach().cpu().numpy()
            transfer_phase = transfer_models[num_frozen].layers[i].phase_mask.phase.detach().cpu().numpy()
            
            # Compute mean absolute difference
            diff = np.mean(np.abs(mnist_phase - transfer_phase))
            phase_diffs[num_frozen].append(diff)
    
    # Plot phase differences
    plt.figure(figsize=(10, 6))
    for num_frozen, diffs in phase_diffs.items():
        plt.plot(range(1, NUM_LAYERS+1), diffs, marker='o', label=f'Freeze {num_frozen} layers')
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Layer')
    plt.ylabel('Mean Absolute Phase Difference')
    plt.title('Phase Mask Differences from Original MNIST Model')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/phase_mask_differences.png')
    
    # Analyze accuracy vs frozen layers
    accuracies = [result['accuracy'] for result in results['fashion_mnist']]
    frozen_counts = [result['frozen_layers'] for result in results['fashion_mnist']]
    
    plt.figure(figsize=(10, 6))
    plt.plot(frozen_counts, accuracies, marker='o')
    plt.xlabel('Number of Frozen Layers')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy vs. Number of Frozen Layers (Fashion-MNIST)')
    plt.grid(True)
    plt.savefig('results/accuracy_vs_frozen_layers.png')
    
    print("\n=== Experiment Completed Successfully ===")
    print(f"Results saved to 'results/' directory")
    print(f"Models saved to 'models/' directory")


if __name__ == "__main__":
    start_time = time()
    run_full_experiment()
    end_time = time()
    print(f"Total runtime: {(end_time - start_time) / 60:.2f} minutes") 