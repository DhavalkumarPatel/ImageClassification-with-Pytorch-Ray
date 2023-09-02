import argparse
import ray
from starlette.requests import Request
from typing import Dict
from io import BytesIO
from fastapi import FastAPI, File, UploadFile

from PIL import Image
import io
import torch

from ray import serve

import evaluate_engine
from torchvision import transforms

@serve.deployment(route_prefix="/", num_replicas="1", ray_actor_options={"num_cpus": 8, "num_gpus": 2})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, experiment_name: str):
        """Initialize the model."""
        self.experiment_name = experiment_name
        self.model = evaluate_engine.return_model(experiment_name)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    async def __call__(self, starlette_request: Request) -> Dict:
        image_payload_bytes = await starlette_request.body()
        pil_image = Image.open(BytesIO(image_payload_bytes))
        print("[1/3] Parsed image data: {}".format(pil_image))

        pil_images = [pil_image]  # Our current batch size is one
        input_tensor = torch.cat(
            [self.preprocessor(i).unsqueeze(0) for i in pil_images]
        )
        print("[2/3] Images transformed, tensor shape {}".format(input_tensor.shape))
        
        self.model.eval()
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        print("[3/3] Inference done!")
        class_index = int(torch.argmax(output_tensor[0]))
        return {"class_index": class_index }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="experiment name to use for serving.")
    args = parser.parse_args()
    ray.init()
    serve.run(ModelDeployment.bind(experiment_name=args.exp_name))
    
    # Use uvicorn to serve the FastAPI app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
    
    