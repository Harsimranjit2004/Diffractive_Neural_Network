"""
Data loader for MNIST dataset and custom images
"""

import numpy as np
import urllib.request
import gzip
import os
from typing import Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from PIL import Image


def load_mnist(data_dir: str = 'data') -> Dict[str, np.ndarray]:
    """
    Load MNIST dataset
    
    Args:
        data_dir: Directory to store/load data
        
    Returns:
        Dictionary with X_train, y_train, X_test, y_test
    """
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # URLs for MNIST dataset
    urls = {
    'train_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    'train_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    'test_images': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
    }

    
    # Local filenames
    filenames = {
        'train_images': os.path.join(data_dir, 'train-images-idx3-ubyte.gz'),
        'train_labels': os.path.join(data_dir, 'train-labels-idx1-ubyte.gz'),
        'test_images': os.path.join(data_dir, 't10k-images-idx3-ubyte.gz'),
        'test_labels': os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    }
    
    # Download files if they don't exist
    for key in urls:
        if not os.path.exists(filenames[key]):
            print(f"Downloading {key}...")
            urllib.request.urlretrieve(urls[key], filenames[key])
    
    # Load train images
    with gzip.open(filenames['train_images'], 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16)
        train_images = train_images.reshape(-1, 28, 28) / 255.0
    
    # Load train labels
    with gzip.open(filenames['train_labels'], 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    # Load test images
    with gzip.open(filenames['test_images'], 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16)
        test_images = test_images.reshape(-1, 28, 28) / 255.0
    
    # Load test labels
    with gzip.open(filenames['test_labels'], 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
    return {
        'X_train': train_images,
        'y_train': train_labels,
        'X_test': test_images,
        'y_test': test_labels
    }


def resize_and_pad_image(image: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """
    Resize and pad image to target size while maintaining aspect ratio
    
    Args:
        image: Input image (H, W) or (H, W, C)
        target_size: Target size (height, width)
        
    Returns:
        Resized and padded image
    """
    # Convert to PIL Image for easier resizing
    if len(image.shape) == 2:
        pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
    else:
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
    
    # Calculate scaling factor to maintain aspect ratio
    orig_w, orig_h = pil_image.size
    scale = min(target_size[0] / orig_h, target_size[1] / orig_w)
    
    # Resize
    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)
    resized_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
    
    # Create empty target image
    if len(image.shape) == 2:
        padded_image = Image.new('L', (target_size[1], target_size[0]), 0)
    else:
        padded_image = Image.new('RGB', (target_size[1], target_size[0]), 0)
    
    # Calculate position to paste (center)
    left = (target_size[1] - new_w) // 2
    top = (target_size[0] - new_h) // 2
    
    # Paste resized image onto target
    padded_image.paste(resized_image, (left, top))
    
    # Convert back to numpy
    result = np.array(padded_image).astype(np.float32) / 255.0
    
    # Ensure result has correct shape
    if len(image.shape) == 2 and len(result.shape) == 3:
        result = result[:, :, 0]  # Take first channel if grayscale
    
    return result


def load_and_process_image(file_path: str, 
                          target_size: Tuple[int, int] = (64, 64), 
                          grayscale: bool = True) -> np.ndarray:
    """
    Load and process an image file
    
    Args:
        file_path: Path to image file
        target_size: Target size (height, width)
        grayscale: Whether to convert to grayscale
        
    Returns:
        Processed image as numpy array
    """
    # Load image
    image = Image.open(file_path)
    
    # Convert to grayscale if needed
    if grayscale and image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array and normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Resize and pad
    return resize_and_pad_image(image_array, target_size)


def prepare_mnist_for_d2nn(dataset: Dict[str, np.ndarray], 
                          target_size: Tuple[int, int] = (64, 64),
                          num_samples: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Prepare MNIST dataset for D2NN processing
    
    Args:
        dataset: MNIST dataset dictionary
        target_size: Target size for images
        num_samples: Number of samples to use (for faster testing)
        
    Returns:
        Dictionary with resized datasets
    """
    result = {}
    
    # Process training data
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    
    if num_samples is not None:
        X_train = X_train[:num_samples]
        y_train = y_train[:num_samples]
    
    # Resize training images
    X_train_resized = np.zeros((len(X_train), *target_size))
    for i, img in enumerate(X_train):
        X_train_resized[i] = resize_and_pad_image(img, target_size)
    
    result['X_train'] = X_train_resized
    result['y_train'] = y_train
    
    # Process test data
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    if num_samples is not None:
        X_test = X_test[:num_samples]
        y_test = y_test[:num_samples]
    
    # Resize test images
    X_test_resized = np.zeros((len(X_test), *target_size))
    for i, img in enumerate(X_test):
        X_test_resized[i] = resize_and_pad_image(img, target_size)
    
    result['X_test'] = X_test_resized
    result['y_test'] = y_test
    
    return result


def visualize_dataset_samples(X: np.ndarray, y: np.ndarray, 
                            num_samples: int = 10) -> plt.Figure:
    """
    Visualize samples from the dataset
    
    Args:
        X: Image data
        y: Labels
        num_samples: Number of samples to visualize
        
    Returns:
        Matplotlib figure
    """
    num_samples = min(num_samples, len(X))
    
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2))
    
    for i in range(num_samples):
        axes[i].imshow(X[i], cmap='gray')
        axes[i].set_title(f"Label: {y[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig