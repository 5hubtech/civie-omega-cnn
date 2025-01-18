import os
import argparse
import json
import torch
from torchvision import transforms
from PIL import Image
from easydict import EasyDict as edict
from model.classifier import Classifier  # noqa
import requests
from io import BytesIO
def load_model(cfg_path, model_path, device):
    # Load configuration
    with open(cfg_path) as f:
        cfg = edict(json.load(f))

    # Initialize model
    model = Classifier(cfg)
    model = torch.nn.DataParallel(model).to(device).eval()
    ckpt = torch.load(model_path, map_location=device)
    model.module.load_state_dict(ckpt['state_dict'])
    return model, cfg

def preprocess_image(image_path, cfg):
    # Define preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((cfg.height, cfg.width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[cfg.pixel_mean / 255.0] * 3, std=[cfg.pixel_std / 255.0] * 3),
    ])
    if image_path.startswith('http://') or image_path.startswith('https://'):
        # Download the image
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimensio

def predict_single_image(cfg, model, image_tensor, device, headers):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)

        # Handle the case where output is a tuple of tensors
        if isinstance(output, tuple):
            # Extract only the prediction tensors (first part of tuple)
            logits = output[0]

            # Stack the tensors along dimension 1 to create a single tensor
            if isinstance(logits, list):
                logits = torch.cat([tensor for tensor in logits], dim=1)

            # Compute probabilities from logits
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
        else:
            # Handle single tensor output
            probabilities = torch.sigmoid(output).cpu().numpy().flatten()

    # Ensure we have the correct number of predictions
    assert len(probabilities) == len(headers), f"Number of predictions ({len(probabilities)}) does not match number of headers ({len(headers)})"

    # Map headers to predictions with full precision
    prediction_dict = {headers[i]: probabilities[i] for i in range(len(headers))}
    return prediction_dict




def main():
    parser = argparse.ArgumentParser(description="Test model with a single image")
    parser.add_argument('--cfg_path', type=str,default="model_files/cfg.json",  help="Path to config file (JSON format)")
    parser.add_argument('--model_path', type=str,default="model_files/best1.ckpt", help="Path to the trained model checkpoint")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and configuration
    model, cfg = load_model(args.cfg_path, args.model_path, device)

    # Define headers (Ensure these match the model's output tasks)
    test_header = [
        'Cardiomegaly',
        'Edema',
        'Consolidation',
        'Atelectasis',
        'Pleural Effusion'
    ]

    # Preprocess the input image
    image_tensor = preprocess_image(args.image_path, cfg)

    # Perform prediction
    prediction_dict = predict_single_image(cfg, model, image_tensor, device, test_header)

    # Display results
    print(f"Predictions for {args.image_path}:")
    for label, prob in prediction_dict.items():
        print(f"  {label}: {prob:.4f}")

if __name__ == '__main__':
    main()

