# For Stable Diffusion
from diffusers import AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import load_image, make_image_grid

# For Flask
from flask import Flask, request, send_file

# Etc.
import torch
import io

# --------------------------------

# Load the Base Image-to-Image Pipeline
# pipeline = AutoPipelineForImage2Image.from_pretrained(
#     "SimianLuo/LCM_Dreamshaper_v7",
#     safety_checker=None
# )

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", variant="fp16", use_safetensors=True)
# controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
pipeline = AutoPipelineForImage2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, variant="fp16", use_safetensors=False
)

# Attach ControlNet to the pipeline
pipeline.controlnet = controlnet


pipeline.to('cpu')
negative_prompt = "Deformed, disfigured, poor details, bad anatomy, dark, colorful"

num_inference_steps = 3

# --------------------------------

app = Flask(__name__)


app.control_image = load_image('james_scare_depth.png')  # Provide a control image file
app.james = load_image('james.png')
@app.route("/generate", methods=["GET"])
def generate_contolled_image():
#     data = request.json
#     prompt = data.get("prompt", "Realistic man on gray background, monochrome")
#     strength = data.get("strength", 0.061)
#     width = data.get("width", 512)
#     height = data.get("height", 512)
    prompt = "add monochrome"

    image_control_net = pipeline(prompt, image=app.james, control_image=app.control_image).images[0]
    make_image_grid([app.james, app.control_image, image_control_net], rows=1, cols=3)
    new_image = pipeline(prompt, negative_prompt=negative_prompt, image=image_control_net, strength=0.45, guidance_scale=10.5).images[0]
    make_image_grid([app.james, app.control_image, image_control_net, new_image], rows=2, cols=2)
    app.james = new_image
    img_io = io.BytesIO()
    new_image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")
    app.james = image
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
