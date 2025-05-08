"""
Optical Physics Module for D2NN Simulator
Contains functions for optical field propagation and manipulation
"""

import numpy as np
from typing import Tuple, Optional


def propagate_fresnel(field: np.ndarray, 
                      dx: float, 
                      wavelength: float, 
                      distance: float) -> np.ndarray:
    """
    Propagate an optical field using Fresnel diffraction
    
    Args:
        field: Complex optical field at input plane
        dx: Pixel size (m)
        wavelength: Wavelength of light (m)
        distance: Propagation distance (m)
    
    Returns:
        Complex optical field at output plane
    """
    # Get field dimensions
    ny, nx = field.shape
    
    # Calculate spatial frequencies
    kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, dx)
    kX, kY = np.meshgrid(kx, ky)
    k = 2 * np.pi / wavelength
    
    # Calculate transfer function
    H = np.exp(1j * distance * np.sqrt(k**2 - kX**2 - kY**2))
    H[kX**2 + kY**2 > k**2] = 0  # Evanescent waves
    
    # Apply propagation in frequency domain
    field_fft = np.fft.fft2(field)
    field_propagated_fft = field_fft * H
    field_propagated = np.fft.ifft2(field_propagated_fft)
    
    return field_propagated


def phase_mask(shape: Tuple[int, int], 
               phase_values: np.ndarray) -> np.ndarray:
    """
    Create a phase-only mask from phase values
    
    Args:
        shape: Shape of the mask (height, width)
        phase_values: Phase values for each pixel (0 to 2π)
    
    Returns:
        Complex transmission function of the phase mask
    """
    # Ensure phase values are in the correct shape
    phase_values = phase_values.reshape(shape)
    
    # Create complex transmission function (magnitude=1, varying phase)
    return np.exp(1j * phase_values)


def intensity(field: np.ndarray) -> np.ndarray:
    """
    Calculate intensity of a complex field
    
    Args:
        field: Complex optical field
        
    Returns:
        Intensity (magnitude squared)
    """
    return np.abs(field)**2


def normalize_field(field: np.ndarray) -> np.ndarray:
    """
    Normalize a complex field to have unit total energy
    
    Args:
        field: Complex optical field
        
    Returns:
        Normalized complex field
    """
    total_energy = np.sum(intensity(field))
    if total_energy > 0:
        return field / np.sqrt(total_energy)
    return field


def visualize_complex_field(field: np.ndarray, 
                           ax=None, 
                           title: Optional[str] = None) -> Tuple:
    """
    Visualize a complex field as magnitude and phase
    
    Args:
        field: Complex optical field
        ax: Optional matplotlib axes (2-element list)
        title: Optional title
        
    Returns:
        Tuple of (magnitude, phase) for plotting
    """
    mag = np.abs(field)
    phase = np.angle(field)
    
    if ax is not None:
        ax[0].imshow(mag, cmap='viridis')
        ax[0].set_title(f"{title} - Magnitude" if title else "Magnitude")
        ax[0].axis('off')
        
        ax[1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax[1].set_title(f"{title} - Phase" if title else "Phase")
        ax[1].axis('off')
    
    return mag, phase


def image_to_field(image: np.ndarray, 
                  amplitude_modulation: bool = True) -> np.ndarray:
    """
    Convert an image to an optical field
    
    Args:
        image: Grayscale image (values between 0 and 1)
        amplitude_modulation: If True, image modulates amplitude, otherwise phase
        
    Returns:
        Complex optical field
    """
    if amplitude_modulation:
        # Image directly modulates amplitude (phase=0)
        return image
    else:
        # Image modulates phase from 0 to 2π
        return np.exp(1j * image * 2 * np.pi)