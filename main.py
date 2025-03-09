from diffusers import AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import load_image, make_image_grid

# For flask
from flask import Flask, request, send_file

# Etc.
import torch
import io

# --------------------------------

# Prepare the SD pipeline
pipe = AutoPipelineForImage2Image.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    safety_checker=None
)
pipe.to('cuda')

# Code for controlnet
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
# controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, variant="fp16", use_safetensors=False, safety_checker=None
)
pipeline.controlnet = controlnet
pipeline.to('cuda')

negative_prompt = "poor details, noise"

# --------------------------------

app = Flask(__name__)

app.james_source = load_image('james.png')
james_prompt ="creepy man"

@app.route("/james.png", methods=["GET"])
def generate_image():
    reset_img = request.args.get('reset', default=False, type=bool)
    strength = request.args.get('str', default=0.1, type=float)
    guidance_scale = request.args.get('gui', default=10.0, type=float)
    num_inference_steps = request.args.get('num', default=5, type=int)
    prompt = request.args.get('prompt', default=james_prompt, type=str)
    count = request.args.get('count', default=1, type=int)

    if reset_img:
        app.james_source = load_image('james.png')

    images = []
    for i in range(count):
        image = pipe(
            image=app.james_source,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=128,
            width=128,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        images.append(image)
        app.james_source = image

    if (count > 1):
        rows = min(count, 3)
        image_grid = make_image_grid(images, rows=rows, cols=3)
        img_io = io.BytesIO()
        image_grid.save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
        

    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

# ----------------

app.control_image = load_image('james_falling.png')  # Provide a control image file
app.control_dst = load_image('james_rah.png')
@app.route("/generate", methods=["GET"])
def generate_contolled_image():
    reset_img = request.args.get('reset', default=False, type=bool)
    strength = request.args.get('str', default=0.1, type=float)
    guidance_scale = request.args.get('gui', default=10.0, type=float)
    num_inference_steps = request.args.get('num', default=5, type=int)
    controlnet_conditioning_scale= request.args.get('num', default=2.0, type=float)
    prompt = request.args.get('prompt', default=james_prompt, type=str)
    prompt = "man with arm raised up"

    new_image = pipeline(
            prompt=prompt,
            image=app.control_dst,
            height=512,
            width=512,
            control_image=app.control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    app.control_dst = new_image

    img_io = io.BytesIO()
    new_image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
