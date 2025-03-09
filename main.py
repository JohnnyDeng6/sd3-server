# For stable diffusion
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from PIL import Image

# For flask
from flask import Flask, request, send_file

# Etc.
import io

# --------------------------------

# Prepare the SD pipeline
pipe = AutoPipelineForImage2Image.from_pretrained(
    "SimianLuo/LCM_Dreamshaper_v7",
    safety_checker=None,
    disable_progress_bar=True
)
pipe.to('cuda')

negative_prompt = "Deformed, disfigured, poor details, bad anatomy, dark, colorful"

# Can be set to 1~50 steps. LCM support fast inference even <= 4 steps. Recommend: 1~8 steps.
num_inference_steps = 2

# --------------------------------

app = Flask(__name__)

app.james = load_image('james.png')
james_prompt = "Realistic man on gray background, monochrome"

@app.route("/james.png", methods=["GET"])
def generate_image():
    image = pipe(
        image=app.james,
        prompt=james_prompt,
        negative_prompt=negative_prompt,
        height=128,
        width=128,
        strength=0.1,
        num_inference_steps=num_inference_steps
    ).images[0]
    app.james = image
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
