import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import time
from scipy.ndimage import rotate


class D2NN:
    """
    Diffractive Deep Neural Network implementation for optical computing simulation
    """
    
    def __init__(self, n_layers=5, layer_size=(200, 200), pixel_size=40e-6, 
                 wavelength=532e-9, layer_distance=40e-3, n_classes=10):
        """
        Initialize a D²NN with specified parameters
        
        Parameters:
            n_layers: Number of diffractive layers
            layer_size: Size of each layer (height, width) in pixels
            pixel_size: Physical size of each pixel in meters
            wavelength: Wavelength of light in meters
            layer_distance: Distance between layers in meters
            n_classes: Number of output classes
        """
        self.n_layers = n_layers
        self.layer_size = layer_size
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.layer_distance = layer_distance
        self.n_classes = n_classes
        
        # Initialize phase masks with random values between 0 and 2π
        self.phase_masks = [2 * np.pi * np.random.random(layer_size) for _ in range(n_layers)]
        
        # Define detector regions for classification
        self.detector_size = (layer_size[0] // n_classes, layer_size[1])
        self.detector_regions = self._create_detector_regions()
    
    def _create_detector_regions(self):
        """Create binary masks for each detector region"""
        regions = []
        height, width = self.layer_size
        region_height = height // self.n_classes
        
        for i in range(self.n_classes):
            region = np.zeros(self.layer_size)
            start_row = i * region_height
            end_row = (i + 1) * region_height
            region[start_row:end_row, :] = 1
            regions.append(region)
        
        return regions
    
    def angular_spectrum_propagation(self, field, propagation_distance):
        """
        Propagate optical field using the angular spectrum method
        
        Parameters:
            field: Complex optical field at the input plane
            propagation_distance: Distance to propagate (m)
        
        Returns:
            Propagated complex optical field
        """
        # Get field dimensions
        ny, nx = field.shape
        
        # Spatial frequencies
        dx = self.pixel_size
        kx = np.fft.fftfreq(nx, dx)
        ky = np.fft.fftfreq(ny, dx)
        kx, ky = np.meshgrid(kx, ky)
        
        # Wave number
        k = 2 * np.pi / self.wavelength
        
        # Transfer function
        kz_squared = k**2 - (2*np.pi*kx)**2 - (2*np.pi*ky)**2
        kz = np.sqrt(np.maximum(0, kz_squared))  # Avoid negative values
        transfer_function = np.exp(1j * kz * propagation_distance)
        
        # Set evanescent waves to zero
        transfer_function[kz_squared < 0] = 0
        
        # Apply propagation in frequency domain
        field_fft = np.fft.fft2(field)
        field_propagated_fft = field_fft * transfer_function
        field_propagated = np.fft.ifft2(field_propagated_fft)
        
        return field_propagated
    
    def forward_pass(self, input_field):
        """
        Perform forward pass through the D²NN
        
        Parameters:
            input_field: Complex input field (amplitude with zero phase)
        
        Returns:
            Complex output field after final layer
        """
        # Initialize with the input field
        field = input_field.copy()
        
        # Propagate through each layer
        for layer_idx in range(self.n_layers):
            # Propagate to the next layer
            field = self.angular_spectrum_propagation(field, self.layer_distance)
            
            # Apply phase modulation of the current layer
            field = field * np.exp(1j * self.phase_masks[layer_idx])
        
        # Final propagation to the detector plane
        output_field = self.angular_spectrum_propagation(field, self.layer_distance)
        
        return output_field
    
    def predict(self, input_image):
        """
        Predict class of input image
        
        Parameters:
            input_image: Input image [height, width] with values in [0, 1]
        
        Returns:
            Predicted class probabilities
        """
        # Convert input image to complex field with uniform phase
        input_field = input_image * np.exp(1j * 0)
        
        # Forward pass through the network
        output_field = self.forward_pass(input_field)
        
        # Calculate intensity at the output plane
        intensity = np.abs(output_field)**2
        
        # Integrate intensity over each detector region
        class_energies = np.array([np.sum(intensity * region) for region in self.detector_regions])
        
        # Convert to probabilities
        probabilities = class_energies / np.sum(class_energies)
        
        return probabilities
    
    def calculate_loss(self, output_prob, target):
        """
        Calculate categorical cross-entropy loss
        
        Parameters:
            output_prob: Predicted probabilities [n_classes]
            target: One-hot encoded target [n_classes]
        
        Returns:
            Loss value
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        output_prob = np.clip(output_prob, epsilon, 1.0 - epsilon)
        
        # Categorical cross-entropy
        loss = -np.sum(target * np.log(output_prob))
        
        return loss
    
    def calculate_gradients(self, input_image, target, layer_idx, delta=0.01):
        """
        Calculate gradients for a specific layer using finite differences
        
        Parameters:
            input_image: Input image
            target: One-hot encoded target
            layer_idx: Index of the layer to calculate gradients for
            delta: Small perturbation value
        
        Returns:
            Gradient matrix for the specified layer
        """
        original_mask = self.phase_masks[layer_idx].copy()
        gradients = np.zeros_like(original_mask)
        
        # Calculate original loss
        original_probs = self.predict(input_image)
        original_loss = self.calculate_loss(original_probs, target)
        
        # Sample random points for gradient estimation
        # For efficiency, we calculate gradients at a subset of points
        sample_rate = 0.05  # Sample 5% of pixels
        sample_mask = np.random.choice([True, False], size=original_mask.shape, 
                                      p=[sample_rate, 1-sample_rate])
        
        y_indices, x_indices = np.where(sample_mask)
        
        for y, x in zip(y_indices, x_indices):
            # Save original value
            original_value = self.phase_masks[layer_idx][y, x]
            
            # Forward perturbation
            self.phase_masks[layer_idx][y, x] = (original_value + delta) % (2 * np.pi)
            forward_probs = self.predict(input_image)
            forward_loss = self.calculate_loss(forward_probs, target)
            
            # Backward perturbation
            self.phase_masks[layer_idx][y, x] = (original_value - delta) % (2 * np.pi)
            backward_probs = self.predict(input_image)
            backward_loss = self.calculate_loss(backward_probs, target)
            
            # Central difference gradient
            gradients[y, x] = (forward_loss - backward_loss) / (2 * delta)
            
            # Restore original value
            self.phase_masks[layer_idx][y, x] = original_value
        
        # For non-sampled points, use interpolation for gradients
        if sample_rate < 1.0:
            from scipy.ndimage import gaussian_filter
            gradients = gaussian_filter(gradients, sigma=1.0)
        
        # Restore the original mask just to be safe
        self.phase_masks[layer_idx] = original_mask
        
        return gradients
    
    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=0.01, 
              momentum=0.9, batch_size=32, epochs=50, frozen_layers=None):
        """
        Train the D²NN using stochastic gradient descent with momentum
        
        Parameters:
            X_train: Training images [n_samples, height, width]
            y_train: One-hot encoded training labels [n_samples, n_classes]
            X_val: Validation images (optional)
            y_val: Validation labels (optional)
            learning_rate: Learning rate for SGD
            momentum: Momentum coefficient
            batch_size: Mini-batch size
            epochs: Number of training epochs
            frozen_layers: List of layer indices to freeze (for transfer learning)
        
        Returns:
            Training history
        """
        n_samples = X_train.shape[0]
        
        # Initialize momentum buffer
        velocity = [np.zeros_like(mask) for mask in self.phase_masks]
        
        # Training history
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        # Define cosine annealing schedule for learning rate
        def cosine_annealing(epoch, total_epochs, initial_lr):
            return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
        
        for epoch in range(epochs):
            # Update learning rate
            current_lr = cosine_annealing(epoch, epochs, learning_rate)
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            epoch_loss = 0
            correct_predictions = 0
            
            start_time = time.time()
            
            # Mini-batch training
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_loss = 0
                batch_gradients = [np.zeros_like(mask) for mask in self.phase_masks]
                
                # Process each sample in the batch
                for idx in batch_indices:
                    # Forward pass
                    output_probs = self.predict(X_train[idx])
                    
                    # Calculate loss and predictions
                    loss = self.calculate_loss(output_probs, y_train[idx])
                    batch_loss += loss
                    
                    if np.argmax(output_probs) == np.argmax(y_train[idx]):
                        correct_predictions += 1
                    
                    # Calculate gradients for each unfrozen layer
                    for layer in range(self.n_layers):
                        if frozen_layers is not None and layer in frozen_layers:
                            continue  # Skip gradient calculation for frozen layers
                        
                        layer_gradients = self.calculate_gradients(X_train[idx], y_train[idx], layer)
                        batch_gradients[layer] += layer_gradients
                
                # Average gradients over the batch
                batch_loss /= len(batch_indices)
                batch_gradients = [grad / len(batch_indices) for grad in batch_gradients]
                
                # Update phase masks using momentum
                for layer in range(self.n_layers):
                    if frozen_layers is not None and layer in frozen_layers:
                        continue  # Skip updates for frozen layers
                    
                    velocity[layer] = momentum * velocity[layer] - current_lr * batch_gradients[layer]
                    self.phase_masks[layer] += velocity[layer]
                    
                    # Ensure phase values stay within [0, 2π]
                    self.phase_masks[layer] = self.phase_masks[layer] % (2 * np.pi)
                
                epoch_loss += batch_loss
            
            # Calculate training metrics
            avg_epoch_loss = epoch_loss / (n_samples // batch_size)
            train_accuracy = correct_predictions / n_samples
            
            history['train_loss'].append(avg_epoch_loss)
            history['train_acc'].append(train_accuracy)
            
            # Calculate validation metrics if validation data is provided
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                validation_str = f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            else:
                validation_str = ""
            
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}, "
                  f"Acc: {train_accuracy:.4f}{validation_str}, Time: {epoch_time:.2f}s")
        
        return history
    
    def evaluate(self, X_test, y_test, batch_size=100):
        """
        Evaluate the model on test data
        
        Parameters:
            X_test: Test images [n_samples, height, width]
            y_test: One-hot encoded test labels [n_samples, n_classes]
            batch_size: Batch size for evaluation
        
        Returns:
            Average loss and accuracy
        """
        n_samples = X_test.shape[0]
        total_loss = 0
        correct_predictions = 0
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            
            for idx in range(start_idx, end_idx):
                output_probs = self.predict(X_test[idx])
                loss = self.calculate_loss(output_probs, y_test[idx])
                total_loss += loss
                
                if np.argmax(output_probs) == np.argmax(y_test[idx]):
                    correct_predictions += 1
        
        avg_loss = total_loss / n_samples
        accuracy = correct_predictions / n_samples
        
        return avg_loss, accuracy
    
    def visualize_fields(self, input_image, layer_indices=None):
        """
        Visualize the complex field at different layers of the network
        
        Parameters:
            input_image: Input image
            layer_indices: Indices of layers to visualize, if None visualize all
        
        Returns:
            Dictionary of complex fields at each requested layer
        """
        if layer_indices is None:
            layer_indices = range(self.n_layers + 1)  # Input + all layers
        
        fields = {}
        
        # Input field
        field = input_image * np.exp(1j * 0)
        
        if 0 in layer_indices:
            fields[0] = field.copy()
        
        # Propagate through each layer
        for layer_idx in range(self.n_layers):
            # Propagate to the next layer
            field = self.angular_spectrum_propagation(field, self.layer_distance)
            
            # Apply phase modulation of the current layer
            field = field * np.exp(1j * self.phase_masks[layer_idx])
            
            if layer_idx + 1 in layer_indices:
                fields[layer_idx + 1] = field.copy()
        
        # Final output field
        if self.n_layers + 1 in layer_indices:
            output_field = self.angular_spectrum_propagation(field, self.layer_distance)
            fields[self.n_layers + 1] = output_field
        
        return fields
    
    def plot_field(self, field, title="Complex Field"):
        """
        Plot a complex field with amplitude as brightness and phase as hue
        
        Parameters:
            field: Complex field to visualize
            title: Plot title
        """
        amplitude = np.abs(field)
        phase = np.angle(field)
        
        # Normalize amplitude to [0, 1]
        norm_amplitude = amplitude / np.max(amplitude)
        
        # Create HSV image: phase as hue, amplitude as value
        hsv = np.zeros((*amplitude.shape, 3))
        hsv[..., 0] = (phase + np.pi) / (2 * np.pi)  # Map phase to [0, 1]
        hsv[..., 1] = 0.9  # Saturation
        hsv[..., 2] = norm_amplitude  # Value
        
        # Convert HSV to RGB
        from matplotlib.colors import hsv_to_rgb
        rgb = hsv_to_rgb(hsv)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(rgb)
        plt.title(title)
        plt.axis('off')
        plt.colorbar(label='Amplitude')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename):
        """Save the model's phase masks to a file"""
        np.savez(filename, *self.phase_masks)
    
    def load_model(self, filename):
        """Load phase masks from a file"""
        data = np.load(filename)
        self.phase_masks = [data[f'arr_{i}'] for i in range(len(data.files))]


def preprocess_mnist(images, labels, n_classes=10, img_size=(200, 200)):
    """
    Preprocess MNIST-like dataset for D²NN
    
    Parameters:
        images: Raw images [n_samples, height*width]
        labels: Raw labels [n_samples]
        n_classes: Number of classes
        img_size: Target image size
    
    Returns:
        Processed images and one-hot encoded labels
    """
    # Reshape images
    n_samples = len(images)
    orig_dim = int(np.sqrt(images.shape[1]))
    images = images.reshape(n_samples, orig_dim, orig_dim)
    
    # Resize images to target size
    from skimage.transform import resize
    resized_images = np.zeros((n_samples, *img_size))
    for i in range(n_samples):
        resized_images[i] = resize(images[i], img_size, anti_aliasing=True)
    
    # Normalize images to [0, 1]
    resized_images = resized_images / np.max(resized_images)
    
    # One-hot encode labels
    lb = LabelBinarizer()
    lb.fit(range(n_classes))
    labels_one_hot = lb.transform(labels.astype(int))
    
    return resized_images, labels_one_hot


def create_rotated_mnist(X, y, angle_range=(-45, 45)):
    """
    Create a rotated version of MNIST dataset
    
    Parameters:
        X: Original images [n_samples, height, width]
        y: Original labels [n_samples, n_classes]
        angle_range: Range of random rotation angles
    
    Returns:
        Rotated images and corresponding labels
    """
    n_samples = X.shape[0]
    rotated_X = np.zeros_like(X)
    
    for i in range(n_samples):
        # Generate random angle within specified range
        angle = np.random.uniform(*angle_range)
        
        # Apply rotation
        rotated_X[i] = rotate(X[i], angle, reshape=False, mode='constant', cval=0.0)
    
    return rotated_X, y


def main():
    """Main function to run the experiment"""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist_images, mnist_labels = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    
    # Load Fashion-MNIST dataset
    print("Loading Fashion-MNIST dataset...")
    fashion_images, fashion_labels = fetch_openml('Fashion-MNIST', version=1, return_X_y=True, parser='auto')
    
    # Convert to numpy arrays
    mnist_images = mnist_images.astype(float).to_numpy()
    mnist_labels = mnist_labels.astype(int).to_numpy()
    fashion_images = fashion_images.astype(float).to_numpy()
    fashion_labels = fashion_labels.astype(int).to_numpy()
    
    # Split data into train and test sets
    X_mnist_train, X_mnist_test, y_mnist_train, y_mnist_test = train_test_split(
        mnist_images, mnist_labels, test_size=0.2, random_state=42
    )
    
    X_fashion_train, X_fashion_test, y_fashion_train, y_fashion_test = train_test_split(
        fashion_images, fashion_labels, test_size=0.2, random_state=42
    )
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    img_size = (200, 200)  # Size for D²NN input
    
    X_mnist_train, y_mnist_train_oh = preprocess_mnist(X_mnist_train, y_mnist_train, img_size=img_size)
    X_mnist_test, y_mnist_test_oh = preprocess_mnist(X_mnist_test, y_mnist_test, img_size=img_size)
    
    X_fashion_train, y_fashion_train_oh = preprocess_mnist(X_fashion_train, y_fashion_train, img_size=img_size)
    X_fashion_test, y_fashion_test_oh = preprocess_mnist(X_fashion_test, y_fashion_test, img_size=img_size)
    
    # Create rotated MNIST dataset
    print("Creating Rotated-MNIST dataset...")
    X_rotated_train, y_rotated_train_oh = create_rotated_mnist(X_mnist_train, y_mnist_train_oh)
    X_rotated_test, y_rotated_test_oh = create_rotated_mnist(X_mnist_test, y_mnist_test_oh)
    
    # Initialize D²NN model
    print("Initializing D²NN model...")
    model = D2NN(n_layers=5, layer_size=img_size, pixel_size=40e-6, 
                wavelength=532e-9, layer_distance=40e-3, n_classes=10)
    
    # Train on MNIST (baseline)
    print("\n--- Training on MNIST (Baseline) ---")
    # Use a smaller subset for demonstration due to computational constraints
    subset_size = 1000  # Adjust based on available computational resources
    history_mnist = model.train(X_mnist_train[:subset_size], y_mnist_train_oh[:subset_size], 
                              X_val=X_mnist_test[:100], y_val=y_mnist_test_oh[:100],
                              learning_rate=0.01, momentum=0.9, batch_size=32, epochs=5)
    
    # Save the MNIST-trained model
    model.save_model('d2nn_mnist.npz')
    
    # Evaluate on MNIST test set
    mnist_loss, mnist_acc = model.evaluate(X_mnist_test[:500], y_mnist_test_oh[:500])
    print(f"MNIST Test Accuracy: {mnist_acc:.4f}")
    
    # Transfer learning to Fashion-MNIST
    print("\n--- Transfer Learning to Fashion-MNIST ---")
    
    # Create a new model with the same configuration
    transfer_model = D2NN(n_layers=5, layer_size=img_size, pixel_size=40e-6, 
                        wavelength=532e-9, layer_distance=40e-3, n_classes=10)
    
    # Load pre-trained weights
    transfer_model.load_model('d2nn_mnist.npz')
    
    # Freeze first two layers and train on Fashion-MNIST
    frozen_layers = [0, 1]  # Freeze first two layers
    history_fashion = transfer_model.train(X_fashion_train[:subset_size], y_fashion_train_oh[:subset_size],
                                        X_val=X_fashion_test[:100], y_val=y_fashion_test_oh[:100],
                                        learning_rate=0.01, momentum=0.9, batch_size=32, epochs=5,
                                        frozen_layers=frozen_layers)
    
    # Evaluate on Fashion-MNIST test set
    fashion_loss, fashion_acc = transfer_model.evaluate(X_fashion_test[:500], y_fashion_test_oh[:500])
    print(f"Fashion-MNIST Transfer Learning Test Accuracy: {fashion_acc:.4f}")
    
    # Transfer learning to Rotated-MNIST
    print("\n--- Transfer Learning to Rotated-MNIST ---")
    
    # Create another new model for rotated MNIST
    rotated_model = D2NN(n_layers=5, layer_size=img_size, pixel_size=40e-6, 
                        wavelength=532e-9, layer_distance=40e-3, n_classes=10)
    
    # Load pre-trained weights
    rotated_model.load_model('d2nn_mnist.npz')
    
    # Freeze only the first layer for Rotated-MNIST
    frozen_layers = [0]  # Freeze only first layer
    history_rotated = rotated_model.train(X_rotated_train[:subset_size], y_rotated_train_oh[:subset_size],
                                        X_val=X_rotated_test[:100], y_val=y_rotated_test_oh[:100],
                                        learning_rate=0.01, momentum=0.9, batch_size=32, epochs=5,
                                        frozen_layers=frozen_layers)
    
    # Evaluate on Rotated-MNIST test set
    rotated_loss, rotated_acc = rotated_model.evaluate(X_rotated_test[:500], y_rotated_test_oh[:500])
    print(f"Rotated-MNIST Transfer Learning Test Accuracy: {rotated_acc:.4f}")
    
    # Compare with training from scratch on Fashion-MNIST
    print("\n--- Training Fashion-MNIST from Scratch ---")
    scratch_fashion_model = D2NN(n_layers=5, layer_size=img_size, pixel_size=40e-6, 
                               wavelength=532e-9, layer_distance=40e-3, n_classes=10)
    
    history_fashion_scratch = scratch_fashion_model.train(X_fashion_train[:subset_size], y_fashion_train_oh[:subset_size],
                                                       X_val=X_fashion_test[:100], y_val=y_fashion_test_oh[:100],
                                                       learning_rate=0.01, momentum=0.9, batch_size=32, epochs=5)
    
    fashion_scratch_loss, fashion_scratch_acc = scratch_fashion_model.evaluate(X_fashion_test[:500], y_fashion_test_oh[:500])
    print(f"Fashion-MNIST From Scratch Test Accuracy: {fashion_scratch_acc:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_fashion['train_acc'], label='Transfer Learning (Train)')
    plt.plot(history_fashion['val_acc'], label='Transfer Learning (Val)')
    plt.plot(history_fashion_scratch['train_acc'], label='From Scratch (Train)')
    plt.plot(history_fashion_scratch['val_acc'], label='From Scratch (Val)')
    plt.title('Fashion-MNIST: Transfer vs Scratch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_fashion['train_loss'], label='Transfer Learning (Train)')
    plt.plot(history_fashion['val_loss'], label='Transfer Learning (Val)')
    plt.plot(history_fashion_scratch['train_loss'], label='From Scratch (Train)')
    plt.plot(history_fashion_scratch['val_loss'], label='From Scratch (Val)')
    plt.title('Fashion-MNIST: Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Visualize optical fields
    print("\n--- Visualizing Optical Fields ---")
    
    # Visualize fields for an MNIST sample
    mnist_sample = X_mnist_test[0]
    mnist_fields = model.visualize_fields(mnist_sample)
    
    # Visualize fields for a Fashion-MNIST sample
    fashion_sample = X_fashion_test[0]
    fashion_fields = transfer_model.visualize_fields(fashion_sample)
    
    # Plot some example fields
    model.plot_field(mnist_fields[1], title="MNIST - Layer 1 Field")
    model.plot_field(mnist_fields[3], title="MNIST - Layer 3 Field")
    
    transfer_model.plot_field(fashion_fields[1], title="Fashion-MNIST - Layer 1 Field")
    transfer_model.plot_field(fashion_fields[3], title="Fashion-MNIST - Layer 3 Field")
    
    # Calculate correlation between MNIST and Fashion-MNIST fields
    correlations = []
    for layer_idx in range(1, model.n_layers + 1):
        mnist_amplitude = np.abs(mnist_fields[layer_idx])
        fashion_amplitude = np.abs(fashion_fields[layer_idx])
        
        # Calculate correlation coefficient
        corr = np.corrcoef(mnist_amplitude.flatten(), fashion_amplitude.flatten())[0, 1]
        correlations.append(corr)
        print(f"Layer {layer_idx} correlation: {corr:.4f}")


if __name__ == "__main__":
    main()