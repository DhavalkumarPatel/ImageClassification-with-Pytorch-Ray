"""
serve_engine Module

This module performs model serving using Gradio and Ray.

References: 
    1. https://docs.ray.io/en/latest/serve/tutorials/gradio-integration.html
    2. https://www.gradio.app/guides/image-classification-in-pytorch
"""

import torch
from torchvision import transforms
import gradio as gr
import evaluate_engine
import os
from ray import serve
from ray.serve.gradio_integrations import GradioServer

def serve_init(experiment_name: str):
    """Initialize the required model and transfroms."""
    
    # Load the model with best checkpoints.
    model = evaluate_engine.return_model(experiment_name)
       
    # Set the model to eval mode.
    model.eval()
    
    # Initialize the transforms.
    preprocess = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    
    return model, preprocess
    
def predict(input_image):
    """Predict the classes with confidence"""
    
    experiment_name = "tuning-resnet-1693589204"
    model, preprocess = serve_init(experiment_name)
    image = preprocess(input_image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(image)
        
    prediction = torch.nn.functional.softmax(logits[0], dim=0)
    
    labels = ['ant', 'bee']
    confidences = {labels[i]: float(prediction[i]) for i in range(2)}
    return confidences

def interface_online():
    """Define the Gradio Interface."""
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=2),
        examples=[os.getcwd()+"/examples/ant.jpg", 
                  os.getcwd()+"/examples/bee.jpg"]).launch()

if __name__ == "__main__":
    app = GradioServer.options(ray_actor_options={"num_cpus": 4})
    serve.run(app.bind(interface_online))
    