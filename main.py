from diffusers import AutoPipelineForImage2Image, ControlNetModel
from diffusers.utils import load_image, make_image_grid

# For flask
from flask import Flask, request, send_file

# Etc.
import torch
import io

import math
from PIL import Image
# --------------------------------

# Prepare the SD pipeline
pipe = AutoPipelineForImage2Image.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    safety_checker=None
)
#pipe.to('cuda')
pipe.to('cpu')

negative_prompt = "poor details, noise"

# --------------------------------

app = Flask(__name__)

img_name = 'test_images/james_to_toast.png'
app.source = load_image(img_name)

@app.route("/generate", methods=["GET"])
def generate_image():
    prompt ="creepy man"
    reset_img = request.args.get('reset', default=False, type=bool) #generate image from original
    strength = request.args.get('str', default=0.1, type=float)
    guidance_scale = request.args.get('gui', default=10.0, type=float)
    num_inference_steps = request.args.get('num', default=5, type=int)
    prompt = request.args.get('prompt', default=prompt, type=str)
    count = request.args.get('count', default=1, type=int) #number of images generated and returned
    width = request.args.get('width', default=128, type=int)
    height = request.args.get('height', default=128, type=int)

    if reset_img:
        app.source = load_image(img_name)

    images = []
    for i in range(count):
        image = pipe(
            image=app.source,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        images.append(image)
        app.source = image

    img_io = io.BytesIO()
    
    if (count > 1):
        size = math.ceil(count ** 0.5)
        while len(images) < size*size:
            blank_image = Image.new("RGB", (width, height), (0, 0, 0))
            images.append(blank_image)
            print(len(images))
        image_grid = make_image_grid(images, rows=size, cols=size)
        image_grid.save(img_io, format="PNG")
    else:
        image.save(img_io, format="PNG")

    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

# ----------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
