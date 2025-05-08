"""
Optimizer module for D2NN training using numerical gradient methods
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from optics import image_to_field, intensity
from d2nn import D2NN


def numerical_gradient(model: D2NN, 
                      input_field: np.ndarray, 
                      target_class: int,
                      epsilon: float = 0.01) -> None:

    # Base loss
    base_loss, _ = model.loss_and_gradient(input_field, target_class)
    
    # For each layer
    for layer_idx, layer in enumerate(model.layers):
        # For each pixel in the layer
        for i in range(layer.shape[0]):
            for j in range(layer.shape[1]):
                # Store original value
                orig_value = layer.phase_values[i, j]
                
                # Perturb forward
                layer.phase_values[i, j] = orig_value + epsilon
                forward_loss, _ = model.loss_and_gradient(input_field, target_class)
                
                # Perturb backward
                layer.phase_values[i, j] = orig_value - epsilon
                backward_loss, _ = model.loss_and_gradient(input_field, target_class)
                
                # Restore original value
                layer.phase_values[i, j] = orig_value
                
                # Central difference gradient
                gradient = (forward_loss - backward_loss) / (2 * epsilon)
                
                # Store gradient
                layer.phase_gradient[i, j] = gradient


def train_model(model: D2NN, 
               X_train: np.ndarray, 
               y_train: np.ndarray,
               learning_rate: float = 0.01,
               batch_size: int = 1,
               epochs: int = 10,
               eval_interval: int = 10) -> Dict[str, List]:

    n_samples = len(X_train)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    history = {
        'loss': [],
        'accuracy': []
    }
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        
        # Shuffle data for this epoch
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        # Process mini-batches
        for batch in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{epochs}"):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, n_samples)
            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]
            
            batch_loss = 0
            
            # Process each sample in the batch
            for X, y in zip(batch_X, batch_y):
                # Convert image to input field
                input_field = image_to_field(X)
                
                # Calculate loss and compute gradients
                loss, predicted = model.loss_and_gradient(input_field, y)
                numerical_gradient(model, input_field, y)
                
                # Update statistics
                batch_loss += loss
                if predicted == y:
                    correct += 1
            
            # Average gradient over the batch
            for layer in model.layers:
                layer.phase_gradient /= (end_idx - start_idx)
            
            # Update parameters
            for layer in model.layers:
                layer.update_parameters(learning_rate)
            
            epoch_loss += batch_loss
        
        # Calculate epoch statistics
        epoch_loss /= n_samples
        accuracy = correct / n_samples
        
        # Store history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(accuracy)
        
        # Print statistics
        if (epoch + 1) % eval_interval == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {accuracy:.4f}")
    
    return history


def evaluate_model(model: D2NN, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray) -> Dict[str, float]:

    n_samples = len(X_test)
    test_loss = 0
    correct = 0
    
    predictions = []
    
    # Process each test sample
    for X, y in tqdm(zip(X_test, y_test), total=n_samples, desc="Evaluating"):
        # Convert image to input field
        input_field = image_to_field(X)
        
        # Forward pass
        loss, predicted = model.loss_and_gradient(input_field, y)
        
        # Update statistics
        test_loss += loss
        if predicted == y:
            correct += 1
        
        predictions.append(predicted)
    
    # Calculate metrics
    test_loss /= n_samples
    accuracy = correct / n_samples
    
    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'predictions': np.array(predictions)
    }


def plot_training_history(history: Dict[str, List]) -> plt.Figure:

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history['loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['accuracy'])
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def visualize_classification(model: D2NN, 
                           X: np.ndarray, 
                           y: int, 
                           figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:

    # Convert image to input field
    input_field = image_to_field(X)
    
    # Forward pass with intermediate results
    result = model.forward(input_field, store_intermediates=True)
    intermediates = result['intermediates']
    
    # Get output intensity and class intensities
    output_intensity = result['output_intensity']
    class_intensities = result['class_intensities']
    predicted_class = np.argmax(class_intensities)
    
    # Create figure
    n_intermediates = len(intermediates)
    n_rows = n_intermediates + 1  # +1 for input and output
    fig = plt.figure(figsize=figsize)
    
    # Plot input image
    ax_input = plt.subplot2grid((n_rows, 2), (0, 0), colspan=2)
    ax_input.imshow(X, cmap='gray')
    ax_input.set_title(f"Input Image (Class {y})")
    ax_input.axis('off')
    
    # Plot intermediate fields
    for i, (name, field) in enumerate(intermediates):
        row = i + 1
        
        # Magnitude subplot
        ax_mag = plt.subplot2grid((n_rows, 2), (row, 0))
        mag = np.abs(field)
        ax_mag.imshow(mag, cmap='viridis')
        ax_mag.set_title(f"{name} - Magnitude")
        ax_mag.axis('off')
        
        # Phase subplot
        ax_phase = plt.subplot2grid((n_rows, 2), (row, 1))
        phase = np.angle(field)
        ax_phase.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title(f"{name} - Phase")
        ax_phase.axis('off')
    
    # Plot output intensity
    ax_output = plt.subplot2grid((n_rows, 2), (n_rows-1, 0), colspan=2)
    im = ax_output.imshow(output_intensity, cmap='inferno')
    ax_output.set_title(f"Output Intensity (Predicted: {predicted_class}, True: {y})")
    
    # Mark detector positions
    for i, (y_pos, x_pos) in enumerate(model.detector_positions):
        color = 'lime' if i == predicted_class else 'red' if i == y else 'white'
        circle = plt.Circle((x_pos, y_pos), 3, color=color, fill=True)
        ax_output.add_patch(circle)
        ax_output.text(x_pos+5, y_pos+5, f"{i}", color='white')
    
    ax_output.axis('off')
    plt.colorbar(im, ax=ax_output, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    return fig