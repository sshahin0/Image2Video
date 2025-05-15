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
    "Horizontal (832x480)": "832*480"
}

# Define task mappings
TASK_MAP = {
    "Text2Video": "t2v-14B",
    "Image2Video": "i2v-14B"
}

def generate_video(image_path, prompt, resolution, task):
    # Save input image temporarily if it's an image2video task
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = os.path.join(OUTPUT_DIR, f"output_{timestamp}.mp4")
    
    command = [
        "python", "generate.py",
        "--task", TASK_MAP[task],
        "--size", resolution,
        "--ckpt_dir", "./Wan2.1-I2V-14B-720P",
        "--prompt", prompt,
        "--offload_model", "true",
        "--t5_cpu",
        "--ulysses_size", "0",
        "--ring_size", "1",
        "--save_file", output_video
    ]

    # Add image parameter only for image2video task
    if task == "Image2Video":
        input_image = os.path.join(OUTPUT_DIR, f"input_{timestamp}.jpg")
        img = Image.open(image_path).convert("RGB")
        img.save(input_image)
        command.extend(["--image", input_image])

    # Run the command
    subprocess.run(command, check=True)

    return output_video

def process(image, prompt, resolution_label, task):
    resolution = RESOLUTION_MAP[resolution_label]
    return generate_video(image, prompt, resolution, task)

# Launch Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("## 🎞️ Wan 2.1 Video Generator")

    with gr.Row():
        with gr.Column(scale=1):
            task = gr.Dropdown(choices=list(TASK_MAP.keys()), value="Image2Video", label="Task")
            image = gr.Image(type="filepath", label="Upload Image", visible=True)
            prompt = gr.Textbox(label="Text Prompt", placeholder="e.g. A fox walking through a snowy forest", lines=2)
            resolution = gr.Dropdown(choices=list(RESOLUTION_MAP.keys()), value="Horizontal (832x480)", label="Resolution")
            run_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Video")

    def update_image_visibility(task):
        return gr.Image.update(visible=(task == "Image2Video"))

    task.change(
        fn=update_image_visibility,
        inputs=[task],
        outputs=[image]
    )

    run_btn.click(
        fn=process,
        inputs=[image, prompt, resolution, task],
        outputs=[output_video]
    )

demo.launch(share=True)
