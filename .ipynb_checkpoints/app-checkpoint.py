from flask import Flask, render_template, request, jsonify
import torch
import base64
import cv2
import numpy as np
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LCMScheduler
import random

# Initialize the Flask application and configurations
app = Flask(__name__)

caption_path = "Salesforce/blip-image-captioning-large"
model_path = "runwayml/stable-diffusion-v1-5"
control_net_path = "lllyasviel/sd-controlnet-canny"
lora_path = "latent-consistency/lcm-lora-sdv1-5"
caption_text = "A drawing of a "
device = "cuda"

# Initialize models outside routes
cap_processor = BlipProcessor.from_pretrained(caption_path)
cap_model = BlipForConditionalGeneration.from_pretrained(caption_path, torch_dtype=torch.float16).to(device)
controlnet = ControlNetModel.from_pretrained(control_net_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
    variant="fp16"
).to(device)

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights(lora_path)

def caption_image(processor, model, image, text="", device="cuda"):
    inputs = processor(image, text, return_tensors="pt").to(device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=20)
    prompt = processor.decode(out[0], skip_special_tokens=True)
    return prompt

def canny_generation(image, t_lower=100, t_upper=200):
    image = np.array(image)
    image = cv2.Canny(image, t_lower, t_upper)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def generate_image(pipe, prompt, controlnet_conditioning_scale, image, num_steps):
    control_image = canny_generation(image)
    
    generator=torch.manual_seed(random.randint(0,1000))
    generated_image = pipe(
        prompt, 
        image=control_image,
        guidance_scale=1, 
        controlnet_conditioning_scale=controlnet_conditioning_scale, 
        cross_attention_kwargs={"scale": 1},
        num_inference_steps=num_steps,
        generator=generator
    ).images[0]
    
    return generated_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transform', methods=['POST'])
def transform_image():
    try:
        data = request.json
        image_data = base64.b64decode(data['imageBase64'])
        image = Image.open(io.BytesIO(image_data))
        controlnet_conditioning_scale = float(data.get('intensityLevel'))
        num_steps = int(data.get('iterationCount', 3))  # Default to 3 if not provided

        # Convert RGBA to RGB with a white background
        if image.mode == 'RGBA':
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            image = background
        image = image.resize((512, 512))
        
        image.save("input_image.png")

        prompt = caption_image(cap_processor, cap_model, image, text=caption_text, device=device)
        print(prompt)
        prompt_words = prompt.split()
        prompt = "HD, 4k, Masterpiece, High Quality art inspired by a " + ' '.join(prompt_words[4:])
        print(prompt)
        result_image = generate_image(pipe, prompt, controlnet_conditioning_scale, image, num_steps=num_steps)  # Pass num_steps here
        result_image.save("result_image.jpg")

        buffered = io.BytesIO()
        result_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        
        return jsonify({'result_image': img_str.decode('utf-8')})

    except Exception as e:
        print(f'error: {str(e)}')
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = 5000
    app.run(debug=True, port=port)
