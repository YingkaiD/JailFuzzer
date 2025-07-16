import io
import sys
import torch
import torch.nn as nn
import uvicorn
from PIL import Image
from einops import rearrange, repeat
import torchvision.transforms as T
from fastapi import Response
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
from pytorch_lightning import seed_everything
import torchvision.transforms as transforms

app = FastAPI()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class ModelWorker():
    def __init__(self):
        self.base = None
        self.refiner = None
        self.g_cuda = torch.Generator(device)
        self.g_cuda.manual_seed(0)
        self.model_load= self.load_pretrained_model()
    
    def load_pretrained_model(self):
        # load both base & refiner
        self.model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True,
                                                             torch_dtype=torch.float16).to(device)
        self.model.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                                    num_train_timesteps=1000)

    def generate_image(self, prompt):

        pil_images = self.model(prompt, num_inference_steps=100, generator=self.g_cuda).images
        pil_image = pil_images[0]

        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr

    
class TextPrompt(BaseModel):
    content_prompt: str

@app.post("/generate-image")
async def generate_image(prompt: TextPrompt):
    prompt = prompt.content_prompt
    img_byte_arr = worker.generate_image(prompt)
    return Response(content=img_byte_arr, media_type="image/png")

if __name__ == '__main__':

    worker = ModelWorker()

    uvicorn.run(app, host="0.0.0.0", port=31001, log_level="info")