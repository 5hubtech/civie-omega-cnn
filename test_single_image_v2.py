# Install required packages
!pip install gradio easydict torch torchvision Pillow requests

import os
import json
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from easydict import EasyDict as edict
import requests
from io import BytesIO
from model.classifier import Classifier  # Make sure this is in your model directory

# Define the headers for classification
HEADERS = [
    'Cardiomegaly',
    'Edema',
    'Consolidation',
    'Atelectasis',
    'Pleural Effusion'
]

def load_model(cfg_path, model_path, device):
    """Load the model and configuration"""
    with open(cfg_path) as f:
        cfg = edict(json.load(f))
    
    model = Classifier(cfg)
    model = torch.nn.DataParallel(model).to(device).eval()
    ckpt = torch.load(model_path, map_location=device)
    model.module.load_state_dict(ckpt['state_dict'])
    return model, cfg

def preprocess_image(image, cfg):
    """Preprocess the input image"""
    transform = transforms.Compose([
        transforms.Resize((cfg.height, cfg.width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[cfg.pixel_mean / 255.0] * 3,
            std=[cfg.pixel_std / 255.0] * 3
        ),
    ])
    
    if isinstance(image, str):
        if image.startswith('http://') or image.startswith('https://'):
            response = requests.get(image)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
    else:
        image = Image.fromarray(image).convert("RGB")
    
    return transform(image).unsqueeze(0)

def predict_single_image(cfg, model, image_tensor, device):
    """Make predictions for a single image"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        
        if isinstance(output, tuple):
            logits = output[0]
            if isinstance(logits, list):
                logits = torch.cat([tensor for tensor in logits], dim=1)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            probabilities = torch.sigmoid(output).cpu().numpy().flatten()
    
    assert len(probabilities) == len(HEADERS)
    return {HEADERS[i]: float(probabilities[i]) for i in range(len(HEADERS))}

# Initialize global variables for model and config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
cfg = None

def initialize_model():
    """Initialize the model (call this before starting Gradio)"""
    global model, cfg
    cfg_path = "model_files/cfg.json"
    model_path = "model_files/best1.ckpt"
    model, cfg = load_model(cfg_path, model_path, device)

def predict_image(input_image):
    """Gradio interface function"""
    if model is None:
        return "Model not initialized. Please initialize the model first."
    
    # Preprocess the image
    image_tensor = preprocess_image(input_image, cfg)
    
    # Get predictions
    predictions = predict_single_image(cfg, model, image_tensor, device)
    
    # Format results for display
    results = []
    for header in HEADERS:
        prob = predictions[header]
        results.append(f"{header}: {prob:.4f}")
    
    return "\n".join(results)

# Create Gradio interface
def create_gradio_interface():
    initialize_model()  # Initialize the model before creating the interface
    
    iface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(),
        outputs=gr.Textbox(label="Predictions"),
        title="Medical Image Classification",
        description="Upload a chest X-ray image to get predictions for various medical conditions.",
        examples=[
            ["sample_image1.jpg"],
            ["sample_image2.jpg"]
        ],
        cache_examples=True
    )
    return iface

# Mount Google Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')

# Create and launch the interface
iface = create_gradio_interface()
iface.launch(share=True)  # share=True creates a public link