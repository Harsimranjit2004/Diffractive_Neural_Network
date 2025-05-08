"""
Main training script for D2NN model
"""

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Import our modules
from data_loader import load_mnist, prepare_mnist_for_d2nn, visualize_dataset_samples
from d2nn import D2NN
from optimizer import train_model, evaluate_model, visualize_classification, plot_training_history

def main():
    parser = argparse.ArgumentParser(description='Train a D2NN model')
    parser.add_argument('--input_size', type=int, default=64, help='Input size (square)')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of D2NN layers')
    parser.add_argument('--layer_distance', type=float, default=0.01, help='Distance between layers (m)')
    parser.add_argument('--pixel_size', type=float, default=1e-5, help='Size of each pixel (m)')
    parser.add_argument('--wavelength', type=float, default=5e-7, help='Wavelength of light (m)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output classes')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples for training')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--save_model', action='store_true', help='Save model parameters')
    parser.add_argument('--load_model', type=str, default=None, help='Load model parameters')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist = load_mnist()
    
    # Prepare dataset for D2NN (resize to target size)
    print(f"Preparing dataset (resizing to {args.input_size}x{args.input_size})...")
    input_shape = (args.input_size, args.input_size)
    dataset = prepare_mnist_for_d2nn(mnist, target_size=input_shape, num_samples=args.num_samples)
    
    # Visualize some samples
    fig = visualize_dataset_samples(dataset['X_train'], dataset['y_train'])
    fig.savefig(os.path.join(args.output_dir, 'dataset_samples.png'))
    plt.close(fig)
    
    # Create D2NN model
    print("Creating D2NN model...")
    model = D2NN(
        input_shape=input_shape,
        num_layers=args.num_layers,
        layer_distance=args.layer_distance,
        pixel_size=args.pixel_size,
        wavelength=args.wavelength,
        num_classes=args.num_classes
    )
    
    # Load model if specified
    if args.load_model is not None and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}...")
        model.load_parameters(args.load_model)
    
    # Train model
    print("Training model...")
    history = train_model(
        model=model,
        X_train=dataset['X_train'],
        y_train=dataset['y_train'],
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        eval_interval=args.eval_interval
    )
    
    # Plot training history
    fig = plot_training_history(history)
    fig.savefig(os.path.join(args.output_dir, 'training_history.png'))
    plt.close(fig)
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, dataset['X_test'], dataset['y_test'])
    
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")
    
    # Visualize some test examples
    num_vis = 5
    for i in range(num_vis):
        idx = np.random.randint(0, len(dataset['X_test']))
        fig = visualize_classification(model, dataset['X_test'][idx], dataset['y_test'][idx])
        fig.savefig(os.path.join(args.output_dir, f'classification_{i}.png'))
        plt.close(fig)
    
    # Save model if specified
    if args.save_model:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output_dir, f'd2nn_model_{timestamp}.npy')
        print(f"Saving model to {model_path}...")
        model.save_parameters(model_path)
        
    print("Done!")


if __name__ == "__main__":
    main()