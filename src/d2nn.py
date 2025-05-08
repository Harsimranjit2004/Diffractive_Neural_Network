"""
Diffractive Deep Neural Network (D2NN) Implementation
Based on "All-Optical Machine Learning Using Diffractive Deep Neural Networks"
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from optics import propagate_fresnel, phase_mask, normalize_field, intensity

class D2NNLayer:
    """
    A single layer in a Diffractive Deep Neural Network
    Implements a phase-only mask with learnable parameters
    """
    def __init__(self, 
                shape: Tuple[int, int], 
                dx: float,
                init_strategy: str = 'random'):
        """
        Initialize a D2NN layer
        
        Args:
            shape: Shape of the layer (height, width)
            dx: Pixel size (m)
            init_strategy: How to initialize phase values ('random', 'zeros')
        """
        self.shape = shape
        self.dx = dx
        
        # Initialize phase values (between 0 and 2π)
        if init_strategy == 'random':
            self.phase_values = np.random.uniform(0, 2*np.pi, shape)
        elif init_strategy == 'zeros':
            self.phase_values = np.zeros(shape)
        else:
            raise ValueError(f"Unknown initialization strategy: {init_strategy}")
        
        # Gradient placeholder
        self.phase_gradient = np.zeros(shape)
    
    def forward(self, input_field: np.ndarray) -> np.ndarray:
        """
        Forward pass through this layer
        
        Args:
            input_field: Complex field incident on this layer
            
        Returns:
            Output field after modulation by the phase mask
        """
        # Apply phase mask to input field
        mask = phase_mask(self.shape, self.phase_values)
        return input_field * mask
    
    def update_parameters(self, learning_rate: float) -> None:
        """
        Update phase values using stored gradients
        
        Args:
            learning_rate: Learning rate for gradient descent
        """
        self.phase_values -= learning_rate * self.phase_gradient
        # Ensure phases stay in [0, 2π] range
        self.phase_values = np.mod(self.phase_values, 2*np.pi)
        
        # Reset gradient for next iteration
        self.phase_gradient = np.zeros(self.shape)


class D2NN:
    """
    Full Diffractive Deep Neural Network with multiple layers
    """
    def __init__(self, 
                input_shape: Tuple[int, int],
                num_layers: int,
                layer_distance: float,
                pixel_size: float,
                wavelength: float,
                num_classes: int):
        """
        Initialize a D2NN
        
        Args:
            input_shape: Shape of input field (height, width)
            num_layers: Number of diffractive layers
            layer_distance: Distance between adjacent layers (m)
            pixel_size: Size of each pixel (m)
            wavelength: Wavelength of light (m)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.layer_distance = layer_distance
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.num_classes = num_classes
        
        # Create layers
        self.layers = [
            D2NNLayer(input_shape, pixel_size) 
            for _ in range(num_layers)
        ]
        
        # Define detector positions for each class
        # Arrange in a grid at the detector plane
        grid_size = int(np.ceil(np.sqrt(num_classes)))
        self.detector_positions = []
        
        center_y, center_x = input_shape[0] // 2, input_shape[1] // 2
        offset = max(input_shape) // (grid_size * 2)
        
        for i in range(num_classes):
            row = i // grid_size
            col = i % grid_size
            y = center_y + (row - grid_size // 2) * offset
            x = center_x + (col - grid_size // 2) * offset
            self.detector_positions.append((y, x))
    
    def forward(self, input_field: np.ndarray, 
                store_intermediates: bool = False) -> Dict[str, Any]:
        """
        Forward pass through the entire network
        
        Args:
            input_field: Complex field at the input plane
            store_intermediates: Whether to store intermediate fields
            
        Returns:
            Dictionary with output field and optional intermediates
        """
        # Normalize input field
        field = normalize_field(input_field)
        
        intermediates = []
        if store_intermediates:
            intermediates.append(('input', field.copy()))
        
        # Propagate through each layer
        for i, layer in enumerate(self.layers):
            # Apply phase mask
            field = layer.forward(field)
            
            if store_intermediates:
                intermediates.append((f'layer_{i}_after_mask', field.copy()))
            
            # Propagate to next layer (except for last layer)
            if i < len(self.layers) - 1:
                field = propagate_fresnel(field, self.pixel_size, 
                                         self.wavelength, self.layer_distance)
                
                if store_intermediates:
                    intermediates.append((f'layer_{i}_after_prop', field.copy()))
        
        # Propagate to detector plane
        output_field = propagate_fresnel(field, self.pixel_size, 
                                        self.wavelength, self.layer_distance)
        
        # Calculate intensities at detector positions
        output_intensities = []
        for y, x in self.detector_positions:
            # Use small area around each detector point
            detector_area = intensity(output_field[y-1:y+2, x-1:x+2])
            output_intensities.append(np.mean(detector_area))
        
        # Calculate intensity over entire output plane
        output_intensity = intensity(output_field)
        
        result = {
            'output_field': output_field,
            'output_intensity': output_intensity,
            'class_intensities': np.array(output_intensities),
        }
        
        if store_intermediates:
            result['intermediates'] = intermediates
        
        return result
    
    def loss_and_gradient(self, input_field: np.ndarray, 
                          target_class: int) -> Tuple[float, np.ndarray]:
        """
        Calculate loss and gradient for backpropagation
        
        Args:
            input_field: Complex field at the input plane
            target_class: Target class index
            
        Returns:
            Tuple of (loss, predicted_class)
        """
        # Forward pass
        result = self.forward(input_field)
        class_intensities = result['class_intensities']
        
        # Calculate loss: want high intensity at target detector, low elsewhere
        target_intensity = class_intensities[target_class]
        other_intensities = np.sum(class_intensities) - target_intensity
        
        # Classification loss - maximize ratio of target to others
        epsilon = 1e-12  # Prevent division by zero
        loss = -np.log(target_intensity / (other_intensities + epsilon) + epsilon)
        
        # Predicted class is the one with highest intensity
        predicted_class = np.argmax(class_intensities)
        
        # For a real backpropagation, we would need to compute the gradients
        # through the optical system, but for simplicity, we'll use numerical
        # gradient approximation in the training loop
        
        return loss, predicted_class
    
    def save_parameters(self, filename: str) -> None:
        """
        Save model parameters to file
        
        Args:
            filename: Output filename
        """
        params = {
            'input_shape': self.input_shape,
            'num_layers': self.num_layers,
            'layer_distance': self.layer_distance,
            'pixel_size': self.pixel_size,
            'wavelength': self.wavelength,
            'num_classes': self.num_classes,
            'detector_positions': self.detector_positions,
            'phase_values': [layer.phase_values for layer in self.layers]
        }
        np.save(filename, params, allow_pickle=True)
    
    def load_parameters(self, filename: str) -> None:
        """
        Load model parameters from file
        
        Args:
            filename: Input filename
        """
        params = np.load(filename, allow_pickle=True).item()
        
        # Ensure the model architecture matches
        assert self.input_shape == params['input_shape']
        assert self.num_layers == params['num_layers']
        
        # Load parameters
        self.layer_distance = params['layer_distance']
        self.pixel_size = params['pixel_size']
        self.wavelength = params['wavelength']
        self.num_classes = params['num_classes']
        self.detector_positions = params['detector_positions']
        
        # Load layer parameters
        for i, phase_values in enumerate(params['phase_values']):
            self.layers[i].phase_values = phase_values