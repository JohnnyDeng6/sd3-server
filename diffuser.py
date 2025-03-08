from flask import Flask, request, send_file
import torch
from diffusers import StableDiffusion3Pipeline
import io
from PIL import Image

app = Flask(__name__)

# Load the Stable Diffusion 3 model
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe.to("cuda")

@app.route("/generate", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt", "")
    negative_prompt = data.get("negative_prompt", "")
    num_inference_steps = data.get("num_inference_steps", 28)
    height = data.get("height", 1024)
    width = data.get("width", 1024)
    guidance_scale = data.get("guidance_scale", 7.0)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
    ).images[0]

    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
