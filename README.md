# Stable Diffusion API with SimianLuo/LCM_Dreamshaper_v7


## Overview
This project provides an API endpoint for generating images using Stable Diffusion with the `SimianLuo/LCM_Dreamshaper_v7` model. It supports image-to-image generation with customizable parameters.


## API Endpoint
`POST /generate`
Generates images based on the given parameters
## Parameters:
- **reset** (`bool`): Set to `True` to generate an image from the original. Default is `False`.
- **str** (`float`): Strength of the image transformation. Controls how much the original image is altered. Default is `0.1`.
- **gui** (`float`): Guidance scale to influence the prompt’s effect. Default is `10.0`.
- **num** (`int`): Number of inference steps to run. Default is `5`.
- **prompt** (`string`): The text prompt to guide the image generation. Default is a predefined prompt.
- **count** (`int`): Number of images to generate. Default is `1`.
- **width** (`int`): Width of the generated image in pixels. Default is `128`.
- **height** (`int`): Height of the generated image in pixels. Default is `128`.

## Example Request:
```bash
GET /generate?prompt=Devil-man&str=0.5&gui=7.5&num=10&count=16&width=512&height=512
```
This will generate 16 devil man images of size 512x512, each with a strength of 0.5, guidance scale of 7.5, and 10 inference steps from the previous. 

<img src="https://github.com/JohnnyDeng6/sd3-server/raw/main/example_image.png" alt="Example Image" width="600"/>




## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/JohnnyDeng6/sd3-server.git
   cd sd3-server
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the server:
  ```bash
 python3 main.py
```

The API will be available at `http://localhost:8888/generate`
