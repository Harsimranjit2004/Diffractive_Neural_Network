"""
Streamlit app for D2NN visualization and interaction
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
import time

# Import our modules
from data_loader import load_mnist, prepare_mnist_for_d2nn, load_and_process_image
from d2nn import D2NN
from optimizer import evaluate_model, visualize_classification
from optics import image_to_field, intensity, visualize_complex_field

# Set page configuration
st.set_page_config(
    page_title="D2NN Optical Neural Network Simulator",
    page_icon="🔍",
    layout="wide"
)

# Sidebar for model configuration
st.sidebar.title("D2NN Configuration")

# Model parameters
input_size = st.sidebar.slider("Input Size", 32, 128, 64, 8)
num_layers = st.sidebar.slider("Number of Layers", 1, 5, 3)
layer_distance = st.sidebar.number_input(
    "Layer Distance (mm)", 
    min_value=0.1, 
    max_value=100.0, 
    value=10.0, 
    step=0.1
) * 1e-3  # Convert to meters
pixel_size = st.sidebar.number_input(
    "Pixel Size (μm)", 
    min_value=0.1, 
    max_value=100.0, 
    value=10.0, 
    step=0.1
) * 1e-6  # Convert to meters
wavelength = st.sidebar.number_input(
    "Wavelength (nm)", 
    min_value=300, 
    max_value=1500, 
    value=500, 
    step=10
) * 1e-9  # Convert to meters
num_classes = st.sidebar.slider("Number of Classes", 2, 10, 10)

# Main content
st.title("Diffractive Deep Neural Network (D2NN) Simulator")
st.markdown("""
This app simulates a Diffractive Deep Neural Network (D2NN) based on the paper 
"All-Optical Machine Learning Using Diffractive Deep Neural Networks" (arXiv:1804.08711v2).

The D2NN uses multiple layers of phase-only masks to classify images through optical diffraction.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Model Visualization", "Test with MNIST", "Upload Image"])

# Model visualization tab
with tab1:
    st.header("D2NN Model Architecture")
    
    # Display model structure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Draw layers
    layer_positions = np.linspace(0, 1, num_layers + 2)  # +2 for input and output
    
    # Input layer
    ax.add_patch(plt.Rectangle((0.1, 0.3), 0.1, 0.4, color='royalblue', alpha=0.8))
    ax.text(0.15, 0.2, "Input", ha='center')
    
    # Diffractive layers
    for i in range(num_layers):
        x_pos = 0.1 + layer_positions[i+1] * 0.8
        ax.add_patch(plt.Rectangle((x_pos, 0.3), 0.05, 0.4, color='green', alpha=0.6))
        ax.text(x_pos + 0.025, 0.2, f"Layer {i+1}", ha='center')
        
        # Draw propagation arrow
        ax.arrow(x_pos - 0.05, 0.5, 0.05 - 0.01, 0, 
                head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Output layer
    ax.add_patch(plt.Rectangle((0.8, 0.3), 0.1, 0.4, color='tomato', alpha=0.8))
    ax.text(0.85, 0.2, "Output", ha='center')
    
    # Draw final propagation arrow
    ax.arrow(0.1 + layer_positions[-2] * 0.8 + 0.05, 0.5, 
            0.8 - (0.1 + layer_positions[-2] * 0.8 + 0.05) - 0.01, 0, 
            head_width=0.02, head_length=0.01, fc='black', ec='black')
    
    # Add annotations for physical parameters
    ax.text(0.5, 0.8, f"λ = {wavelength*1e9:.1f} nm, Δz = {layer_distance*1e3:.1f} mm, Δx = {pixel_size*1e6:.1f} μm",
            ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    st.pyplot(fig)
    
    st.markdown("""
    ### Key Components:
    - **Input Layer**: The input image (e.g., MNIST digit) is converted to an optical field
    - **Diffractive Layers**: Each layer is a phase-only mask that diffracts light
    - **Propagation**: The optical field propagates between layers via Fresnel diffraction
    - **Output Layer**: The detector plane captures the intensity pattern, with specific regions assigned to each class
    """)
    
    # Theoretical background
    st.subheader("Theoretical Background")
    st.markdown("""
    #### Fresnel Diffraction:
    The propagation of light between layers is modeled using Fresnel diffraction:
    
    $$U_2(x_2,y_2) = \\frac{e^{jkz}}{j\\lambda z} \\int\\int U_1(x_1,y_1) e^{\\frac{jk}{2z}[(x_2-x_1)^2+(y_2-y_1)^2]} dx_1 dy_1$$
    
    where $U_1$ is the field at the input plane, $U_2$ is the field at the output plane, $k=2\\pi/\\lambda$ is the wave number, and $z$ is the propagation distance.
    
    #### Phase Modulation:
    Each diffractive layer modulates the phase of the incoming light:
    
    $$t(x,y) = e^{j\\phi(x,y)}$$
    
    where $\\phi(x,y)$ represents the learnable phase values at each pixel.
    """)

# Test with MNIST tab
with tab2:
    st.header("Test with MNIST Digits")
    
    # Create/load model and dataset
    @st.cache_resource
    def load_model_and_data(_input_size, _num_layers, _layer_distance, _pixel_size, _wavelength, _num_classes):
        # Create model
        model = D2NN(
            input_shape=(_input_size, _input_size),
            num_layers=_num_layers,
            layer_distance=_layer_distance,
            pixel_size=_pixel_size,
            wavelength=_wavelength,
            num_classes=_num_classes
        )
        
        # Try to load a pre-trained model if it exists
        model_path = os.path.join('results', 'd2nn_model.npy')
        if os.path.exists(model_path):
            try:
                model.load_parameters(model_path)
                st.sidebar.success("Loaded pre-trained model")
            except:
                st.sidebar.warning("Failed to load pre-trained model, using random initialization")
        else:
            st.sidebar.info("No pre-trained model found, using random initialization")
        
        # Load a small subset of MNIST for testing
        mnist = load_mnist()
        dataset = prepare_mnist_for_d2nn(mnist, target_size=(_input_size, _input_size), num_samples=100)
        
        return model, dataset
    
    model, dataset = load_model_and_data(input_size, num_layers, layer_distance, pixel_size, wavelength, num_classes)
    
    # Display some test samples
    st.subheader("MNIST Test Samples")
    cols = st.columns(5)
    selected_indices = []
    
    for i, col in enumerate(cols):
        idx = np.random.randint(0, len(dataset['X_test']))
        selected_indices.append(idx)
        col.image(dataset['X_test'][idx], caption=f"Digit: {dataset['y_test'][idx]}", use_column_width=True)
    
    # Select a digit to visualize
    selected_idx = st.selectbox("Select a digit to visualize", options=range(len(selected_indices)),
                              format_func=lambda i: f"Digit {dataset['y_test'][selected_indices[i]]}")
    
    idx = selected_indices[selected_idx]
    img = dataset['X_test'][idx]
    label = dataset['y_test'][idx]
    
    # Run forward pass through the model
    st.subheader("D2NN Propagation")
    with st.spinner("Running optical propagation..."):
        # Convert image to input field
        input_field = image_to_field(img)
        
        # Forward pass with intermediate results
        result = model.forward(input_field, store_intermediates=True)
        intermediates = result['intermediates']
        
        # Get output intensity and class intensities
        output_intensity = result['output_intensity']
        class_intensities = result['class_intensities']
        predicted_class = np.argmax(class_intensities)
        
        # Create visualization of all fields
        n_intermediates = len(intermediates)
        
        # Display input and prediction
        st.markdown(f"**Input: MNIST Digit {label}, Predicted: {predicted_class}**")
        
        # Create tabs for each layer
        layer_tabs = st.tabs(["Input"] + [f"Layer {i+1}" for i in range(num_layers)] + ["Output"])
        
        # Input tab
        with layer_tabs[0]:
            cols = st.columns(2)
            with cols[0]:
                st.image(img, caption="Input Image", use_column_width=True)
            with cols[1]:
                # Convert to complex field
                mag, phase = visualize_complex_field(input_field)
                st.image(mag, caption="Input Field Magnitude", use_column_width=True)
        
        # Layer tabs
        for i in range(num_layers):
            with layer_tabs[i+1]:
                cols = st.columns(2)
                # After mask
                with cols[0]:
                    field_after_mask = intermediates[i*2+1][1]
                    mag, phase = visualize_complex_field(field_after_mask)
                    st.image(mag, caption=f"Layer {i+1} - After Mask (Magnitude)", use_column_width=True)
                    st.image(phase, caption=f"Layer {i+1} - After Mask (Phase)", clamp=True, use_column_width=True)
                
                # After propagation (if not the last layer)
                if i < num_layers - 1:
                    with cols[1]:
                        field_after_prop = intermediates[i*2+2][1]
                        mag, phase = visualize_complex_field(field_after_prop)
                        st.image(mag, caption=f"Layer {i+1} - After Propagation (Magnitude)", use_column_width=True)
                        st.image(phase, caption=f"Layer {i+1} - After Propagation (Phase)", clamp=True, use_column_width=True)
        
        # Output tab
        with layer_tabs[-1]:
            # Output field
            cols = st.columns(2)
            with cols[0]:
                output_field = result['output_field']
                mag, phase = visualize_complex_field(output_field)
                st.image(mag, caption="Output Field Magnitude", use_column_width=True)
            
            # Output intensity with detector positions
            with cols[1]:
                # Create a figure for the output intensity
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(output_intensity, cmap='inferno')
                ax.set_title(f"Output Intensity (Predicted: {predicted_class}, True: {label})")
                
                # Mark detector positions
                for i, (y_pos, x_pos) in enumerate(model.detector_positions):
                    color = 'lime' if i == predicted_class else 'red' if i == label else 'white'
                    circle = plt.Circle((x_pos, y_pos), 3, color=color, fill=True)
                    ax.add_patch(circle)
                    ax.text(x_pos+5, y_pos+5, f"{i}", color='white')
                
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
            
            # Display class intensities
            bar_data = {str(i): intensity for i, intensity in enumerate(class_intensities)}
            st.bar_chart(bar_data)

# Upload custom image tab
with tab3:
    st.header("Test with Your Own Image")
    
    uploaded_file = st.file_uploader("Upload a grayscale image (ideally a handwritten digit)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load and process the uploaded image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        # Handle different image types
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Convert RGB to grayscale
            image = image.convert('L')
            st.info("Converted color image to grayscale")
        
        # Process image to correct size and format
        processed_img = load_and_process_image(uploaded_file, target_size=(input_size, input_size))
        
        # Display original and processed images
        cols = st.columns(2)
        with cols[0]:
            st.image(image, caption="Original Image", use_column_width=True)
        with cols[1]:
            st.image(processed_img, caption="Processed Image", use_column_width=True)
        
        # Run forward pass
        if st.button("Run D2NN Classification"):
            with st.spinner("Processing..."):
                # Create/get model
                model, _ = load_model_and_data(input_size, num_layers, layer_distance, pixel_size, wavelength, num_classes)
                
                # Convert image to input field
                input_field = image_to_field(processed_img)
                
                # Forward pass with intermediate results
                result = model.forward(input_field, store_intermediates=True)
                
                # Get predicted class
                class_intensities = result['class_intensities']
                predicted_class = np.argmax(class_intensities)
                
                # Display prediction
                st.success(f"Prediction: Digit {predicted_class}")
                
                # Display output intensity
                output_intensity = result['output_intensity']
                fig, ax = plt.subplots(figsize=(6, 6))
                im = ax.imshow(output_intensity, cmap='inferno')
                ax.set_title(f"Output Intensity (Predicted: {predicted_class})")
                
                # Mark detector positions
                for i, (y_pos, x_pos) in enumerate(model.detector_positions):
                    color = 'lime' if i == predicted_class else 'white'
                    circle = plt.Circle((x_pos, y_pos), 3, color=color, fill=True)
                    ax.add_patch(circle)
                    ax.text(x_pos+5, y_pos+5, f"{i}", color='white')
                
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)
                
                # Display class intensities
                st.subheader("Class Prediction Intensities")
                bar_data = {str(i): intensity for i, intensity in enumerate(class_intensities)}
                st.bar_chart(bar_data)

# Add explanatory information
st.sidebar.markdown("---")
st.sidebar.subheader("About D2NN")
st.sidebar.markdown("""
A Diffractive Deep Neural Network (D2NN) uses multiple layers of diffractive elements to perform 
machine learning tasks entirely through optical diffraction. 

Unlike traditional neural networks that use electronic processing, D2NNs leverage the wave nature of light
to perform computation at the speed of light.

**References**:
- Lin, X. et al. "All-Optical Machine Learning Using Diffractive Deep Neural Networks" (Science, 2018)
- ArXiv: 1804.08711v2
""")

# Workflow information
st.sidebar.markdown("---")
st.sidebar.subheader("Project Workflow")
st.sidebar.markdown("""
1. **optics.py**: Core optical propagation functions
2. **d2nn.py**: Diffractive neural network implementation
3. **data_loader.py**: Data loading and preprocessing
4. **optimizer.py**: Training and evaluation functions
5. **train.py**: Main training script
6. **app.py**: This Streamlit interface
""")
st.sidebar.markdown("To train a model, run: `python train.py`")
