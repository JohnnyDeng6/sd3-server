from diffusers import AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import load_image, make_image_grid

# For flask
from flask import Flask, request, send_file

# Etc.
import torch
import io

# --------------------------------

# Prepare the SD pipeline
# pipe = AutoPipelineForImage2Image.from_pretrained(
#     "SimianLuo/LCM_Dreamshaper_v7",
#     safety_checker=None
# )
# pipe.to('cuda')

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
james_prompt = "man on gray background, 8k"

@app.route("/james.png", methods=["GET"])
def generate_image():
    reset_img = request.args.get('reset', default=False, type=bool)
    strength = request.args.get('str', default=0.1, type=float)
    guidance_scale = request.args.get('gui', default=10.0, type=float)
    num_inference_steps = request.args.get('num', default=5, type=int)
    prompt = request.args.get('prompt', default=james_prompt, type=str)

    if reset_img:
        app.james_source = load_image('james.png')

    image = pipeline(
        image=app.james_source,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=128,
        width=128,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    app.james_source = image

    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

# ----------------

app.control_image = load_image('james_scare_depth.png')  # Provide a control image file
app.control_dst = load_image('james.png')
@app.route("/generate", methods=["GET"])
def generate_contolled_image():
#     data = request.json
#     prompt = data.get("prompt", "Realistic man on gray background, monochrome")
#     strength = data.get("strength", 0.061)
#     width = data.get("width", 512)
#     height = data.get("height", 512)
    prompt = "add monochrome"

    new_image = pipeline(prompt, image=app.control_dst, control_image=app.control_image).images[0]
    #new_image = pipeline(prompt, negative_prompt=negative_prompt, image=image_control_net, strength=0.45, guidance_scale=10.5).images[0]
    app.control_dst = new_image

    img_io = io.BytesIO()
    new_image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
