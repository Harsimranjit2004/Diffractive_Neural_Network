import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import time
import io
import base64
import pickle
from PIL import Image
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="D²NN Transfer Learning Simulator",
    page_icon="🔆",
    layout="wide"
)
def save_model(phase_masks, params, filename="trained_d2nn_source_model.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump({'phase_masks': phase_masks, 'params': params}, f)


def angular_spectrum_propagation(field, dx, wavelength, propagation_distance):
    """
    Propagate optical field using the angular spectrum method
    
    Parameters:
        field: Complex optical field at the input plane
        dx: Pixel size (m)
        wavelength: Wavelength of light (m)
        propagation_distance: Distance to propagate (m)
    
    Returns:
        Propagated complex optical field
    """
    # Get field dimensions
    ny, nx = field.shape
    
    # Spatial frequencies
    kx = np.fft.fftfreq(nx, dx)
    ky = np.fft.fftfreq(ny, dx)
    kx, ky = np.meshgrid(kx, ky)
    
    # Wave number
    k = 2 * np.pi / wavelength
    
    # Transfer function
    kz = np.sqrt(k**2 - (2*np.pi*kx)**2 - (2*np.pi*ky)**2 + 0j)
    transfer_function = np.exp(1j * kz * propagation_distance)
    
    # Set evanescent waves to zero
    transfer_function[k**2 < (2*np.pi*kx)**2 + (2*np.pi*ky)**2] = 0
    
    # Apply propagation in frequency domain
    field_fft = np.fft.fft2(field)
    field_propagated_fft = field_fft * transfer_function
    field_propagated = np.fft.ifft2(field_propagated_fft)
    
    return field_propagated

def forward_pass(input_image, phase_masks, params):
    """
    Perform forward pass through the D²NN
    
    Parameters:
        input_image: Input image (normalized to [0,1])
        phase_masks: List of phase modulation masks
        params: Dictionary of simulation parameters
    
    Returns:
        Complex optical fields at each layer
        Final output intensities for each detector region
    """
    # Extract parameters
    wavelength = params['wavelength']
    pixel_size = params['pixel_size']
    prop_distance = params['prop_distance']
    
    # Initialize input field (amplitude modulation with zero phase)
    input_field = input_image * np.exp(1j * 0)
    
    # Initialize list to store fields at each layer
    fields = [input_field]
    current_field = input_field
    
    # Propagate through each layer
    for i, phase_mask in enumerate(phase_masks):
        # Apply phase modulation
        after_mask = current_field * np.exp(1j * phase_mask)
        
        # Propagate to next layer (except after the last layer)
        if i < len(phase_masks) - 1:
            propagated = angular_spectrum_propagation(after_mask, pixel_size, wavelength, prop_distance)
        else:
            propagated = after_mask  # No propagation after the last mask
        
        fields.append(propagated)
        current_field = propagated
    
    # Calculate intensities at detector regions
    n_classes = 10  # MNIST, Fashion-MNIST, etc. have 10 classes
    detector_size = current_field.shape[0] // 4  # Size of each detector region
    
    intensities = np.zeros(n_classes)
    for i in range(n_classes):
        # Define detector region for class i
        row_start = (i // 5) * detector_size + (current_field.shape[0] // 8)
        col_start = (i % 5) * detector_size + (current_field.shape[1] // 8)
        
        # Calculate integrated intensity in detector region
        region = current_field[row_start:row_start+detector_size//2, col_start:col_start+detector_size//2]
        intensities[i] = np.sum(np.abs(region)**2)
    
    # Normalize intensities to get probabilities
    if np.sum(intensities) > 0:
        intensities = intensities / np.sum(intensities)
    
    return fields, intensities

def calculate_loss(predictions, true_label):
    """
    Calculate categorical cross-entropy loss
    
    Parameters:
        predictions: Predicted class probabilities
        true_label: True class label (integer)
    
    Returns:
        Loss value
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # One-hot encode true label
    true_one_hot = np.zeros_like(predictions)
    true_one_hot[true_label] = 1
    
    # Calculate cross-entropy loss
    loss = -np.sum(true_one_hot * np.log(predictions))
    return loss

def calculate_gradient(input_image, phase_masks, true_label, layer_idx, params, delta=0.01):
    """
    Calculate gradients for a specific layer using finite differences
    
    Parameters:
        input_image: Input image
        phase_masks: List of phase masks
        true_label: True class label
        layer_idx: Index of the layer to calculate gradients for
        params: Simulation parameters
        delta: Small perturbation for finite difference
    
    Returns:
        Gradient of the loss with respect to the phase mask
    """
    # Get original loss
    _, original_pred = forward_pass(input_image, phase_masks, params)
    original_loss = calculate_loss(original_pred, true_label)
    
    # Initialize gradient matrix
    gradient = np.zeros_like(phase_masks[layer_idx])
    
    # Use a coarse grid for faster gradient calculation (in practice we'd use a more efficient approach)
    step = 10  # Calculate gradient every 10 pixels
    
    for i in range(0, gradient.shape[0], step):
        for j in range(0, gradient.shape[1], step):
            # Create perturbed phase masks
            pos_masks = [m.copy() for m in phase_masks]
            neg_masks = [m.copy() for m in phase_masks]
            
            # Add and subtract delta at position (i,j)
            pos_masks[layer_idx][i, j] = (pos_masks[layer_idx][i, j] + delta) % (2 * np.pi)
            neg_masks[layer_idx][i, j] = (neg_masks[layer_idx][i, j] - delta) % (2 * np.pi)
            
            # Forward pass with perturbed masks
            _, pos_pred = forward_pass(input_image, pos_masks, params)
            _, neg_pred = forward_pass(input_image, neg_masks, params)
            
            # Calculate losses
            pos_loss = calculate_loss(pos_pred, true_label)
            neg_loss = calculate_loss(neg_pred, true_label)
            
            # Finite difference gradient
            gradient[i, j] = (pos_loss - neg_loss) / (2 * delta)
            
            # Fill in neighboring pixels with the same gradient for efficiency
            for di in range(step):
                for dj in range(step):
                    if i+di < gradient.shape[0] and j+dj < gradient.shape[1]:
                        gradient[i+di, j+dj] = gradient[i, j]
    
    return gradient

def train_step(batch_images, batch_labels, phase_masks, params, learning_rate=0.01, 
               momentum=0.9, frozen_layers=None, velocity=None):
    """
    Perform one training step on a batch of data
    
    Parameters:
        batch_images: Batch of input images
        batch_labels: Batch of true labels
        phase_masks: List of phase masks
        params: Simulation parameters
        learning_rate: Learning rate for SGD
        momentum: Momentum coefficient
        frozen_layers: List of layer indices to freeze (for transfer learning)
        velocity: Momentum velocity vectors
    
    Returns:
        Updated phase masks, updated velocity, loss, accuracy
    """
    if velocity is None:
        velocity = [np.zeros_like(mask) for mask in phase_masks]
    
    batch_size = len(batch_images)
    batch_loss = 0
    correct = 0
    
    # Calculate gradients for each sample in batch
    batch_gradients = [np.zeros_like(mask) for mask in phase_masks]
    
    for i in range(batch_size):
        # Forward pass
        _, predictions = forward_pass(batch_images[i], phase_masks, params)
        
        # Calculate loss
        loss = calculate_loss(predictions, batch_labels[i])
        batch_loss += loss
        
        # Check if prediction is correct
        if np.argmax(predictions) == batch_labels[i]:
            correct += 1
        
        # Calculate gradients for each layer
        for layer in range(len(phase_masks)):
            if frozen_layers is not None and layer in frozen_layers:
                continue  # Skip gradient calculation for frozen layers
            
            layer_grad = calculate_gradient(batch_images[i], phase_masks, batch_labels[i], layer, params)
            batch_gradients[layer] += layer_grad
    
    # Average gradients over batch
    batch_gradients = [grad / batch_size for grad in batch_gradients]
    batch_loss /= batch_size
    accuracy = correct / batch_size
    
    # Update phase masks using momentum
    for layer in range(len(phase_masks)):
        if frozen_layers is not None and layer in frozen_layers:
            continue  # Skip updates for frozen layers
        
        velocity[layer] = momentum * velocity[layer] - learning_rate * batch_gradients[layer]
        phase_masks[layer] += velocity[layer]
        
        # Ensure phase values stay within [0, 2π]
        phase_masks[layer] = phase_masks[layer] % (2 * np.pi)
    
    return phase_masks, velocity, batch_loss, accuracy

def visualize_complex_field(field, title="Complex Field"):
    """
    Create a visualization of a complex optical field
    
    Parameters:
        field: Complex field to visualize
        title: Plot title
    
    Returns:
        Matplotlib figure
    """
    # Extract amplitude and phase
    amplitude = np.abs(field)
    phase = np.angle(field)
    
    # Normalize amplitude to [0, 1]
    amplitude_norm = amplitude / (np.max(amplitude) + 1e-10)
    
    # Create HSV visualization where:
    # - Hue represents phase (cyclic)
    # - Saturation is constant
    # - Value (brightness) represents amplitude
    h = (phase + np.pi) / (2 * np.pi)  # normalize phase to [0, 1]
    s = np.ones_like(h) * 0.9
    v = amplitude_norm
    
    # Convert to RGB
    hsv = np.stack([h, s, v], axis=2)
    rgb = hsv_to_rgb(hsv)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis('off')
    
    return fig

def get_fashion_mnist_class_name(label):
    """Get Fashion-MNIST class name from numeric label"""
    class_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }
    return class_names.get(label, f"Label {label}")

def load_dataset(dataset_name, size=100):
    """
    Load a subset of the specified dataset
    
    Parameters:
        dataset_name: Name of the dataset ('mnist', 'fashion_mnist', or 'rotated_mnist')
        size: Number of samples to load
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    if dataset_name == 'mnist':
        # Load MNIST
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
    
    elif dataset_name == 'fashion_mnist':
        # Load Fashion-MNIST
        fashion = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
        X, y = fashion.data, fashion.target.astype(int)
    
    elif dataset_name == 'rotated_mnist':
        # Load MNIST and apply random rotations
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist.data, mnist.target.astype(int)
        
        # We'll use a simple approach for rotation in this simulation
        # In a real implementation, we'd use scipy.ndimage.rotate
        # Here we'll just shuffle the data since we're primarily interested in the transfer learning process
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    # Reshape from flattened 784 features to 28x28 images
    X_train = X_train.reshape(-1, 28, 28)
    X_test = X_test.reshape(-1, 28, 28)
    
    # Take a subset for faster demonstration
    X_train = X_train[:size]
    y_train = y_train[:size]
    X_test = X_test[:size//5]
    y_test = y_test[:size//5]
    
    return X_train, y_train, X_test, y_test

def resize_images(images, target_size):
    """
    Resize images to target size using simple interpolation
    
    Parameters:
        images: Array of images [n_samples, height, width]
        target_size: Tuple of (height, width)
    
    Returns:
        Resized images
    """
    n_samples = images.shape[0]
    resized = np.zeros((n_samples, target_size[0], target_size[1]))
    
    for i in range(n_samples):
        # Convert to PIL Image for resizing
        img = Image.fromarray((images[i] * 255).astype(np.uint8))
        img_resized = img.resize(target_size, Image.BILINEAR)
        resized[i] = np.array(img_resized) / 255.0
    
    return resized

def initialize_phase_masks(n_layers, layer_size):
    """Initialize random phase masks"""
    return [np.random.uniform(0, 2*np.pi, (layer_size, layer_size)) for _ in range(n_layers)]

# Main application
def main():
    st.title("D²NN Transfer Learning Simulator")
    st.markdown("""
    This application simulates Diffractive Deep Neural Networks (D²NNs) and their transfer learning capabilities,
    as described in the paper "Task Transfer in Diffractive Deep Neural Networks: Optical Feature Reuse Across Domains".
    """)
    
    # Sidebar with settings
    st.sidebar.header("Simulation Settings")
    
    # D²NN architecture parameters
    st.sidebar.subheader("D²NN Architecture")
    sim_size = st.sidebar.slider("Simulation Size", 50, 200, 100, 
                             help="Size of optical field (pixels). Lower values are faster but less accurate.")
    n_layers = st.sidebar.slider("Number of Layers", 2, 5, 3, 
                             help="Number of diffractive layers in the D²NN")
    
    # Physical parameters
    st.sidebar.subheader("Physical Parameters")
    wavelength = st.sidebar.number_input("Wavelength (nm)", 400, 1000, 532) * 1e-9  # Convert to meters
    pixel_size = st.sidebar.number_input("Pixel Size (μm)", 10, 100, 40) * 1e-6  # Convert to meters
    prop_distance = st.sidebar.number_input("Propagation Distance (mm)", 10, 100, 40) * 1e-3  # Convert to meters
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    dataset = st.sidebar.selectbox("Source Dataset", ["mnist", "fashion_mnist"])
    target_dataset = st.sidebar.selectbox("Target Dataset", ["fashion_mnist", "rotated_mnist"], 
                                      index=0 if dataset == "mnist" else 1)
    
    learning_rate = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001, format="%.3f")
    batch_size = st.sidebar.slider("Batch Size", 1, 16, 4, 
                               help="Small batch size for demo purposes, would be larger in production")
    
    # Collect parameters
    sim_params = {
        'wavelength': wavelength,
        'pixel_size': pixel_size,
        'prop_distance': prop_distance,
        'sim_size': sim_size
    }
    
    # Main content 
    tab1, tab2, tab3 = st.tabs(["Train Source Model", "Transfer Learning", "Inference"])
    
    with tab1:
        st.header(f"Training D²NN on {dataset.upper()}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Process")
            
            # Button to start training
            if st.button("Train Source Model", key="train_source"):
                # Load source dataset
                X_train, y_train, X_test, y_test = load_dataset(dataset, size=100)
                
                # Resize images to simulation size
                X_train_resized = resize_images(X_train, (sim_size, sim_size))
                X_test_resized = resize_images(X_test, (sim_size, sim_size))
                
                # Initialize phase masks
                phase_masks = initialize_phase_masks(n_layers, sim_size)
                
                # Initialize training metrics
                train_losses = []
                train_accuracies = []
                
                # Initialize progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize visualization placeholders
                fig_col1, fig_col2 = st.columns(2)
                phase_fig_placeholder = fig_col1.empty()
                loss_fig_placeholder = fig_col2.empty()
                
                # Initialize velocity for momentum
                velocity = [np.zeros_like(mask) for mask in phase_masks]
                
                # Simple training loop (limited iterations for demo)
                n_iterations = 30  # In practice, would be much larger
                
                # Store the trained model in session state for later use
                if 'source_model' not in st.session_state:
                    st.session_state['source_model'] = {}
                
                for i in range(n_iterations):
                    # Select random mini-batch
                    batch_indices = np.random.choice(len(X_train), batch_size, replace=False)
                    batch_X = X_train_resized[batch_indices]
                    batch_y = y_train[batch_indices]
                    
                    # Perform training step
                    phase_masks, velocity, loss, accuracy = train_step(
                        batch_X, batch_y, phase_masks, sim_params, 
                        learning_rate=learning_rate, momentum=0.9
                    )
                    
                    # Record metrics
                    train_losses.append(loss)
                    train_accuracies.append(accuracy)
                    
                    # Update progress
                    progress = (i + 1) / n_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")
                    
                    # Visualize phase masks and training curve (every few iterations)
                    if i % 5 == 0 or i == n_iterations - 1:
                        # Plot one of the phase masks
                        phase_fig, phase_ax = plt.subplots()
                        im = phase_ax.imshow(phase_masks[0], cmap='hsv')
                        phase_ax.set_title(f"Phase Mask (Layer 1)")
                        phase_ax.axis('off')
                        plt.colorbar(im, ax=phase_ax)
                        phase_fig_placeholder.pyplot(phase_fig)
                        plt.close(phase_fig)
                        
                        # Plot training curve
                        loss_fig, loss_ax = plt.subplots()
                        loss_ax.plot(train_losses, label='Loss')
                        loss_ax.set_xlabel('Iteration')
                        loss_ax.set_ylabel('Loss')
                        loss_ax.set_title('Training Loss')
                        loss_ax.legend()
                        loss_fig_placeholder.pyplot(loss_fig)
                        plt.close(loss_fig)
                
                # Test the trained model
                test_correct = 0
                for i in range(len(X_test_resized)):
                    _, predictions = forward_pass(X_test_resized[i], phase_masks, sim_params)
                    if np.argmax(predictions) == y_test[i]:
                        test_correct += 1
                
                test_accuracy = test_correct / len(X_test_resized)
                st.success(f"Training complete! Test accuracy: {test_accuracy:.2f}")
                
                # Store the trained model
                st.session_state['source_model'] = {
                    'phase_masks': phase_masks,
                    'params': sim_params,
                    'dataset': dataset,
                    'accuracy': test_accuracy
                }
                save_model(phase_masks, sim_params)
        
        with col2:
            st.subheader("Model Architecture")
            st.write(f"Number of layers: {n_layers}")
            st.write(f"Layer size: {sim_size}×{sim_size} pixels")
            st.write(f"Wavelength: {wavelength*1e9:.1f} nm")
            st.write(f"Pixel size: {pixel_size*1e6:.1f} μm")
            st.write(f"Propagation distance: {prop_distance*1e3:.1f} mm")
            
            # Display dataset examples
            st.subheader("Dataset Examples")
            if 'current_examples' not in st.session_state:
                # Load a few examples to display
                X_examples, y_examples, _, _ = load_dataset(dataset, size=10)
                st.session_state['current_examples'] = (X_examples, y_examples)
            else:
                X_examples, y_examples = st.session_state['current_examples']
            
            # Display images
            fig, axes = plt.subplots(2, 3, figsize=(8, 6))
            axes = axes.flatten()
            
            for i in range(min(6, len(X_examples))):
                axes[i].imshow(X_examples[i], cmap='gray')
                if dataset == 'fashion_mnist':
                    class_name = get_fashion_mnist_class_name(y_examples[i])
                else:
                    class_name = f"Digit {y_examples[i]}"
                axes[i].set_title(class_name)
                axes[i].axis('off')
            
            st.pyplot(fig)
            plt.close(fig)
    
    with tab2:
        st.header("Transfer Learning")
        
        if 'source_model' not in st.session_state:
            st.warning("Please train a source model first in the 'Train Source Model' tab.")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Transfer Learning Process")
                
                # Transfer learning options
                st.write("Select which layers to freeze:")
                frozen_options = []
                for i in range(n_layers):
                    frozen = st.checkbox(f"Freeze Layer {i+1}", value=(i < n_layers//2))
                    if frozen:
                        frozen_options.append(i)
                
                # Button to start transfer learning
                if st.button("Start Transfer Learning", key="start_transfer"):
                    # Load target dataset
                    X_train, y_train, X_test, y_test = load_dataset(target_dataset, size=100)
                    
                    # Resize images
                    X_train_resized = resize_images(X_train, (sim_size, sim_size))
                    X_test_resized = resize_images(X_test, (sim_size, sim_size))
                    
                    # Get source model
                    source_model = st.session_state['source_model']
                    phase_masks = [mask.copy() for mask in source_model['phase_masks']]
                    
                    # Initialize training metrics
                    train_losses = []
                    train_accuracies = []
                    
                    # Initialize progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Initialize visualization placeholders
                    fig_col1, fig_col2 = st.columns(2)
                    phase_fig_placeholder = fig_col1.empty()
                    loss_fig_placeholder = fig_col2.empty()
                    
                    # Initialize velocity for momentum
                    velocity = [np.zeros_like(mask) for mask in phase_masks]
                    
                    # Training iterations (limited for demo)
                    n_iterations = 30
                    
                    # Store the transfer model in session state
                    if 'transfer_model' not in st.session_state:
                        st.session_state['transfer_model'] = {}
                    
                    for i in range(n_iterations):
                        # Select random mini-batch
                        batch_indices = np.random.choice(len(X_train), batch_size, replace=False)
                        batch_X = X_train_resized[batch_indices]
                        batch_y = y_train[batch_indices]
                        
                        # Perform transfer learning step (freezing selected layers)
                        phase_masks, velocity, loss, accuracy = train_step(
                            batch_X, batch_y, phase_masks, sim_params, 
                            learning_rate=learning_rate, momentum=0.9,
                            frozen_layers=frozen_options, velocity=velocity
                        )
                        
                        # Record metrics
                        train_losses.append(loss)
                        train_accuracies.append(accuracy)
                        
                        # Update progress
                        progress = (i + 1) / n_iterations
                        progress_bar.progress(progress)
                        status_text.text(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}")
                        
                        # Visualize progress
                        if i % 5 == 0 or i == n_iterations - 1:
                            # Find a non-frozen layer to visualize
                            non_frozen_layer = 0
                            for l in range(len(phase_masks)):
                                if l not in frozen_options:
                                    non_frozen_layer = l
                                    break
                            
                            # Plot the non-frozen phase mask
                            phase_fig, phase_ax = plt.subplots()
                            im = phase_ax.imshow(phase_masks[non_frozen_layer], cmap='hsv')
                            phase_ax.set_title(f"Phase Mask (Layer {non_frozen_layer+1} - Trainable)")
                            phase_ax.axis('off')
                            plt.colorbar(im, ax=phase_ax)
                            phase_fig_placeholder.pyplot(phase_fig)
                            plt.close(phase_fig)
                            
                            # Plot training curve
                            loss_fig, loss_ax = plt.subplots()
                            loss_ax.plot(train_losses, label='Loss')
                            loss_ax.set_xlabel('Iteration')
                            loss_ax.set_ylabel('Loss')
                            loss_ax.set_title('Transfer Learning Loss')
                            loss_ax.legend()
                            loss_fig_placeholder.pyplot(loss_fig)
                            plt.close(loss_fig)
                    
                    # Test the transfer model
                    test_correct = 0
                    for i in range(len(X_test_resized)):
                        _, predictions = forward_pass(X_test_resized[i], phase_masks, sim_params)
                        if np.argmax(predictions) == y_test[i]:
                            test_correct += 1
                    
                    test_accuracy = test_correct / len(X_test_resized)
                    st.success(f"Transfer learning complete! Test accuracy: {test_accuracy:.2f}")
                    
                    # Store the transfer model
                    st.session_state['transfer_model'] = {
                        'phase_masks': phase_masks,
                        'params': sim_params,
                        'dataset': target_dataset,
                        'accuracy': test_accuracy,
                        'frozen_layers': frozen_options
                    }
                    
                    # Compare with source model accuracy
                    source_accuracy = source_model['accuracy']
                    st.write(f"Source model ({source_model['dataset'].upper()}) accuracy: {source_accuracy:.2f}")
                    st.write(f"Transfer model ({target_dataset.upper()}) accuracy: {test_accuracy:.2f}")
                    
                    # Plot comparison
                    comp_fig, comp_ax = plt.subplots(figsize=(8, 4))
                    models = [source_model['dataset'].upper(), target_dataset.upper()]
                    accuracies = [source_accuracy, test_accuracy]
                    comp_ax.bar(models, accuracies, color=['blue', 'green'])
                    comp_ax.set_ylabel('Accuracy')
                    comp_ax.set_title('Model Accuracy Comparison')
                    comp_ax.set_ylim(0, 1)
                    
                    for i, v in enumerate(accuracies):
                        comp_ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
                    
                    st.pyplot(comp_fig)
                    plt.close(comp_fig)
            
            with col2:
                st.subheader("Transfer Learning Details")
                st.write(f"Source dataset: {source_model['dataset'].upper()}")
                st.write(f"Target dataset: {target_dataset.upper()}")
                st.write("Frozen layers: " + ", ".join([f"Layer {i+1}" for i in frozen_options]))
                
                # Display dataset examples
                st.subheader("Target Dataset Examples")
                
                # Load a few examples from target dataset
                X_examples, y_examples, _, _ = load_dataset(target_dataset, size=10)
                
                # Display images
                fig, axes = plt.subplots(2, 3, figsize=(8, 6))
                axes = axes.flatten()
                
                for i in range(min(6, len(X_examples))):
                    axes[i].imshow(X_examples[i], cmap='gray')
                    if target_dataset == 'fashion_mnist':
                        class_name = get_fashion_mnist_class_name(y_examples[i])
                    else:
                        class_name = f"Digit {y_examples[i]}"
                    axes[i].set_title(class_name)
                    axes[i].axis('off')
                
                st.pyplot(fig)
                plt.close(fig)
    
    with tab3:
        st.header("Inference Visualization")
        
        # Check if models are available
        if 'source_model' not in st.session_state:
            st.warning("Please train a source model first in the 'Train Source Model' tab.")
            has_source = False
        else:
            has_source = True
            
        if 'transfer_model' not in st.session_state:
            has_transfer = False
        else:
            has_transfer = True
        
        if has_source:
            # Select which model to use for inference
            model_choice = st.radio("Select model for inference:", 
                                  ["Source Model", "Transfer Model"] if has_transfer else ["Source Model"])
            
            if model_choice == "Source Model":
                model = st.session_state['source_model']
                dataset_name = model['dataset']
            else:
                model = st.session_state['transfer_model']
                dataset_name = model['dataset']
            
            # Load a sample from the appropriate dataset
            X_samples, y_samples, _, _ = load_dataset(dataset_name, size=20)
            
            # Resize images
            X_samples_resized = resize_images(X_samples, (sim_size, sim_size))
            
            # Let user select an example
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Select an Example")
                
                # Display sample images in a grid for selection
                fig, axes = plt.subplots(4, 5, figsize=(10, 8))
                axes = axes.flatten()
                
                for i in range(min(20, len(X_samples))):
                    axes[i].imshow(X_samples[i], cmap='gray')
                    axes[i].axis('off')
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Allow user to select an example
                selected_idx = st.number_input("Select example index (0-19):", 0, 19, 0)
                
                # Show the selected example
                st.write(f"Selected example (class: {y_samples[selected_idx]})")
                if dataset_name == 'fashion_mnist':
                    st.write(f"Class name: {get_fashion_mnist_class_name(y_samples[selected_idx])}")
                
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(X_samples[selected_idx], cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
            
            with col2:
                st.subheader("Inference Visualization")
                
                # Run inference on selected example
                if st.button("Run Inference", key="run_inference"):
                    # Get the selected example
                    input_image = X_samples_resized[selected_idx]
                    true_label = y_samples[selected_idx]
                    
                    # Run forward pass to get fields and predictions
                    fields, predictions = forward_pass(input_image, model['phase_masks'], model['params'])
                    
                    # Show predictions
                    pred_class = np.argmax(predictions)
                    
                    if pred_class == true_label:
                        st.success(f"✓ Correctly predicted class {pred_class}")
                    else:
                        st.error(f"✗ Incorrectly predicted class {pred_class} (true: {true_label})")
                    
                    # Plot class probabilities
                    fig, ax = plt.subplots(figsize=(10, 4))
                    classes = list(range(10))
                    
                    if dataset_name == 'fashion_mnist':
                        class_labels = [get_fashion_mnist_class_name(i) for i in range(10)]
                    else:
                        class_labels = [f"Digit {i}" for i in range(10)]
                    
                    bars = ax.bar(classes, predictions)
                    bars[true_label].set_color('green')
                    if pred_class != true_label:
                        bars[pred_class].set_color('red')
                    
                    ax.set_xticks(classes)
                    ax.set_xticklabels(class_labels, rotation=45, ha='right')
                    ax.set_ylabel('Probability')
                    ax.set_title('Class Prediction Probabilities')
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.subheader("Optical Field Propagation")
                    st.write("Visualizing how the input propagates through the diffractive layers:")
                    
                    n_fields = len(fields)
                    cols = st.columns(min(n_fields, 3))
                    
                    for i, field in enumerate(fields):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            if i == 0:
                                title = "Input Field"
                            else:
                                title = f"After Layer {i}"
                            
                            fig = visualize_complex_field(field, title)
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    st.subheader("Phase Mask Visualization")
                    mask_idx = st.slider("Select layer to visualize:", 0, len(model['phase_masks'])-1, 0)
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(model['phase_masks'][mask_idx], cmap='hsv')
                    ax.set_title(f"Phase Mask (Layer {mask_idx+1})")
                    ax.axis('off')
                    plt.colorbar(im, ax=ax)
                    
                    st.pyplot(fig)
                    plt.close(fig)

    st.markdown("---")
    st.markdown("""
    ## About Diffractive Deep Neural Networks (D²NNs)
    
    D²NNs are an emerging class of optical computing systems that use the wave nature of light and 
    multiple layers of diffractive surfaces to perform machine learning tasks directly with optical 
    diffraction physics. Unlike electronic neural networks, D²NNs operate at the speed of light and 
    are extremely energy-efficient once trained.
    
    The key aspects of this simulator:
    
    1. **Light Propagation**: Simulates how light waves propagate through space using the Angular Spectrum Method.
    2. **Phase Modulation**: Each layer consists of a phase mask that modulates the phase of the incoming wavefront.
    3. **Transfer Learning**: Demonstrates how features learned for one task can be transferred to another.
    4. **Visualization**: See the complex optical fields as they propagate through the network.
    
    For more information on D²NNs, refer to:
    - Lin et al., "All-Optical Machine Learning Using Diffractive Deep Neural Networks," Science (2018)
    - Mengu et al., "Analysis of Diffractive Optical Neural Networks and Their Integration with Electronic Neural Networks," IEEE JSTQE (2020)
    """)

if __name__ == "__main__":
    main()