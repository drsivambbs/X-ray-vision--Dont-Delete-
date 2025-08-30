import gradio as gr
import torchxrayvision as xrv
import torch
import numpy as np
from PIL import Image

# Load pretrained model
model = xrv.models.DenseNet(weights="densenet121-res224-all")
model.eval()

def diagnose_xray(image):
    """Diagnose X-ray image using TorchXRayVision"""
    if image is None:
        return "Please upload an X-ray image"
    
    # Preprocess image
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = xrv.datasets.normalize(img, 255)  # Normalize
    
    # Resize to model input size
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0).float()
    img = torch.nn.functional.interpolate(img, size=(224, 224))
    
    # Get predictions
    with torch.no_grad():
        outputs = model(img)
        predictions = torch.sigmoid(outputs).cpu().numpy()[0]
    
    # Create results dictionary
    results = {}
    for i, pathology in enumerate(model.pathologies):
        results[pathology] = float(predictions[i])
    
    return results

# Create Gradio interface
demo = gr.Interface(
    fn=diagnose_xray,
    inputs=gr.Image(type="pil", label="Upload X-ray Image"),
    outputs=gr.Label(num_top_classes=10, label="Diagnosis Predictions"),
    title="🩻 X-ray Diagnosis Tool",
    description="Upload an X-ray image to get AI-powered diagnosis predictions using TorchXRayVision",
    examples=[],
    api_name="diagnose"
)

if __name__ == "__main__":
    demo.launch(share=True)