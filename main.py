# For stable diffusion
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image

# For flask
from flask import Flask, request, send_file

# Etc.
import io

# --------------------------------

# Prepare the SD pipeline
pipe = AutoPipelineForImage2Image.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    safety_checker=None
)
pipe.to('cuda')

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
        app.james = load_image('james.png')

    image = pipe(
        image=app.james,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=128,
        width=128,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    ).images[0]
    app.james = image

    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
