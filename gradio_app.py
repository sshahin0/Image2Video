import gradio as gr
import os
from datetime import datetime
from PIL import Image
import subprocess

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define resolution mappings
RESOLUTION_MAP = {
    "720p (1280x720)": "1280*720",
    "Horizontal (832x480)": "832*480",
    "Square (1024x1024)": "1024*1024"
}

def generate_video(image_path, prompt, resolution):
    # Save input image temporarily
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_image = os.path.join(OUTPUT_DIR, f"input_{timestamp}.jpg")
    output_video = os.path.join(OUTPUT_DIR, f"i2v_{timestamp}.mp4")

    img = Image.open(image_path).convert("RGB")
    img.save(input_image)

    # Construct the command
    command = [
        "python", "generate.py",
        "--task", "i2v-14B",
        "--size", resolution,
        "--ckpt_dir", "./Wan2.1-I2V-14B-720P",
        "--image", input_image,
        "--prompt", prompt,
        "--offload_model", "true",
        "--t5_cpu",
        "--ulysses_size", "0",
        "--ring_size", "1",
        "--save_file", output_video
    ]

    # Run the command
    subprocess.run(command, check=True)

    return output_video

def process(image, prompt, resolution_label):
    resolution = RESOLUTION_MAP[resolution_label]
    return generate_video(image, prompt, resolution)

# Launch Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎞️ Wan 2.1 Image-to-Video Generator")

    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(type="filepath", label="Upload Image")
            prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. A fox walking through a snowy forest", lines=2)
            resolution = gr.Dropdown(choices=list(RESOLUTION_MAP.keys()), value="Horizontal (832x480)", label="Resolution")
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video")

    run_btn.click(
        fn=process,
        inputs=[image, prompt, resolution],
        outputs=[output_video]
    )

demo.launch(share=True)
