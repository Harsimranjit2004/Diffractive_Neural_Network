import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from optics import propagate_fresnel, phase_mask, normalize_field, intensity

class D2NNLayer:

    def __init__(self, 
                shape: Tuple[int, int], 
                dx: float,
                init_strategy: str = 'random'):
      
        self.shape = shape
        self.dx = dx
        
        if init_strategy == 'random':
            self.phase_values = np.random.uniform(0, 2*np.pi, shape)
        elif init_strategy == 'zeros':
            self.phase_values = np.zeros(shape)
        else:
            raise ValueError(f"Unknown initialization strategy: {init_strategy}")
        
        self.phase_gradient = np.zeros(shape)
    
    def forward(self, input_field: np.ndarray) -> np.ndarray:
       
        mask = phase_mask(self.shape, self.phase_values)
        return input_field * mask
    
    def update_parameters(self, learning_rate: float) -> None:
       
        self.phase_values -= learning_rate * self.phase_gradient
        self.phase_values = np.mod(self.phase_values, 2*np.pi)
        
        self.phase_gradient = np.zeros(self.shape)


class D2NN:

    def __init__(self, 
                input_shape: Tuple[int, int],
                num_layers: int,
                layer_distance: float,
                pixel_size: float,
                wavelength: float,
                num_classes: int):
   
        self.input_shape = input_shape
        self.num_layers = num_layers
        self.layer_distance = layer_distance
        self.pixel_size = pixel_size
        self.wavelength = wavelength
        self.num_classes = num_classes
        
        self.layers = [
            D2NNLayer(input_shape, pixel_size) 
            for _ in range(num_layers)
        ]
        
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

        field = normalize_field(input_field)
        
        intermediates = []
        if store_intermediates:
            intermediates.append(('input', field.copy()))
        
        for i, layer in enumerate(self.layers):
            field = layer.forward(field)
            
            if store_intermediates:
                intermediates.append((f'layer_{i}_after_mask', field.copy()))
            
            if i < len(self.layers) - 1:
                field = propagate_fresnel(field, self.pixel_size, 
                                         self.wavelength, self.layer_distance)
                
                if store_intermediates:
                    intermediates.append((f'layer_{i}_after_prop', field.copy()))
        
        output_field = propagate_fresnel(field, self.pixel_size, 
                                        self.wavelength, self.layer_distance)
        
        output_intensities = []
        for y, x in self.detector_positions:
            detector_area = intensity(output_field[y-1:y+2, x-1:x+2])
            output_intensities.append(np.mean(detector_area))
        
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
  
        result = self.forward(input_field)
        class_intensities = result['class_intensities']
        
        target_intensity = class_intensities[target_class]
        other_intensities = np.sum(class_intensities) - target_intensity
        
        epsilon = 1e-12  # Prevent division by zero
        loss = -np.log(target_intensity / (other_intensities + epsilon) + epsilon)
        
        predicted_class = np.argmax(class_intensities)
        
        
        return loss, predicted_class
    
    def save_parameters(self, filename: str) -> None:
       
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

        params = np.load(filename, allow_pickle=True).item()
        
        assert self.input_shape == params['input_shape']
        assert self.num_layers == params['num_layers']
        
        self.layer_distance = params['layer_distance']
        self.pixel_size = params['pixel_size']
        self.wavelength = params['wavelength']
        self.num_classes = params['num_classes']
        self.detector_positions = params['detector_positions']
        
        for i, phase_values in enumerate(params['phase_values']):
            self.layers[i].phase_values = phase_values