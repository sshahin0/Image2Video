import gradio as gr
import os
from datetime import datetime
from PIL import Image
import subprocess
from pillow_heif import register_heif_opener
import logging
import sys
import time
import huggingface_hub
from pathlib import Path

# Set up logging to show in IDE
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wan_generation.log')
    ]
)
logger = logging.getLogger(__name__)

# Register HEIF opener with PIL
register_heif_opener()

OUTPUT_DIR = "outputs"
LORA_DIR = "lora_weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

# Define resolution mappings
RESOLUTION_MAP = {
    "720p (1280x720)": "1280*720",
    "Horizontal (832x480)": "832*480",
    "Vertical (480x832)": "480*832"
}

# Define task mappings and their corresponding checkpoint directories
TASK_MAP = {
    "Text2Video": "t2v-14B",
    "Image2Video": "i2v-14B"
}

CHECKPOINT_MAP = {
    "Text2Video": "./Wan2.1-T2V-14B",
    "Image2Video": "./Wan2.1-I2V-14B-720P"
}

def download_lora_weights(repo_id, filename=None):
    """Download LoRA weights from Hugging Face Hub"""
    try:
        logger.info(f"Downloading LoRA weights from {repo_id}")
        if filename:
            local_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=LORA_DIR,
                local_dir_use_symlinks=False
            )
        else:
            # Download all files from the repo
            local_path = huggingface_hub.snapshot_download(
                repo_id=repo_id,
                local_dir=os.path.join(LORA_DIR, repo_id.split('/')[-1]),
                local_dir_use_symlinks=False
            )
        logger.info(f"LoRA weights downloaded to {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Error downloading LoRA weights: {str(e)}")
        return None

def convert_video_for_gradio(input_path):
    """Convert video to a format that Gradio can handle using system ffmpeg."""
    try:
        # Create a new filename for the converted video
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}_gradio.mp4"
        logger.info(f"Converting video for Gradio: {input_path} -> {output_path}")
        
        # Use system ffmpeg to convert the video with more compatible settings
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(output_path):
            logger.info("Video conversion completed successfully")
            # Ensure the video is readable by Gradio
            with open(output_path, 'rb') as f:
                f.read(1)  # Test if file is readable
            return output_path
        logger.error("Video conversion failed")
        return input_path
        
    except Exception as e:
        logger.error(f"Error in convert_video_for_gradio: {str(e)}")
        return input_path

def process_with_status(image, prompt, resolution_label, task, num_inference_steps, seed, flow_shift, 
                       fps, speed, sample_guiding_scale, lora_weights, lora_scale):
    try:
        start_time = time.time()  # Start timing
        # Map the resolution label to the actual value
        resolution = RESOLUTION_MAP[resolution_label]
        logger.info(f"Starting video generation - Task: {task}, Resolution: {resolution}")
        
        # For Text2Video task, image parameter is not needed
        if task == "Text2Video":
            image = None
            logger.info("Text-to-Video generation selected")
        elif task == "Image2Video" and (image is None or not os.path.exists(image)):
            logger.error("No image provided for Image-to-Video generation")
            return None, "Error: Please upload an image for Image-to-Video generation"
            
        # Generate video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_video = os.path.abspath(os.path.join(OUTPUT_DIR, f"output_{timestamp}.mp4"))
        logger.info(f"Output video will be saved to: {output_video}")
        
        command = [
            "python", "generate.py",
            "--task", TASK_MAP[task],
            "--size", resolution,
            "--ckpt_dir", CHECKPOINT_MAP[task],
            "--prompt", prompt,
            "--offload_model", "true",
            "--t5_cpu",
            "--ulysses_size", "0",
            "--ring_size", "1",
            "--save_file", output_video,
            "--sample_steps", str(num_inference_steps),
            "--base_seed", str(int(seed)),
            "--sample_shift", str(float(flow_shift)),
            "--fps", str(int(fps)),
            "--speed", str(float(speed)),
            "--sample_guiding_scale", str(float(sample_guiding_scale))
        ]

        # Handle LoRA weights
        if lora_weights:
            if lora_weights.startswith("hf://"):
                # Extract repo_id and filename from the path
                path_parts = lora_weights[5:].split("/")
                repo_id = "/".join(path_parts[:2])
                filename = path_parts[2] if len(path_parts) > 2 else None
                
                # Download LoRA weights
                local_path = download_lora_weights(repo_id, filename)
                if local_path:
                    command.extend(["--lora_weights", local_path])
                    command.extend(["--lora_scale", str(float(lora_scale))])
                    logger.info(f"Using LoRA weights from {local_path}")
                else:
                    logger.warning("Failed to download LoRA weights, proceeding without LoRA")
            elif os.path.exists(lora_weights):
                command.extend(["--lora_weights", lora_weights])
                command.extend(["--lora_scale", str(float(lora_scale))])
                logger.info(f"Using local LoRA weights from {lora_weights}")
            else:
                logger.warning(f"LoRA weights path {lora_weights} not found, proceeding without LoRA")

        # Add image parameter only for image2video task
        if task == "Image2Video":
            input_image = os.path.abspath(os.path.join(OUTPUT_DIR, f"input_{timestamp}.jpg"))
            img = Image.open(image).convert("RGB")
            img.save(input_image)
            command.extend(["--image", input_image])
            command.extend(["--frame_num", "81"])  # Set frame_num to 81 for image2video task
            logger.info(f"Input image saved to: {input_image}")

        logger.info(f"Running command: {' '.join(command)}")
        
        # Run the command and capture output
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output in real-time
        output = []
        step_count = 0
        total_steps = num_inference_steps
        last_progress = ""
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.strip()
                output.append(line)
                logger.info(f"Generation progress: {line}")
                
                # Try to extract step information
                if "step" in line.lower():
                    try:
                        step_count = int(line.split("step")[1].split("/")[0].strip())
                        progress = (step_count / total_steps) * 100
                        elapsed_time = time.time() - start_time
                        status = f"Generation progress: {progress:.1f}% ({step_count}/{total_steps} steps)\nElapsed time: {elapsed_time:.1f}s\n{line}"
                    except:
                        status = line
                else:
                    status = line
                
                yield None, status
        
        if process.returncode != 0:
            error = process.stderr.read()
            logger.error(f"Generation failed: {error}")
            return None, f"Error: {error}"
        
        if os.path.exists(output_video):
            logger.info("Video file generated successfully")
            yield None, "Video file generated successfully. Converting format..."
            # Convert video to Gradio-compatible format
            converted_video = convert_video_for_gradio(output_video)
            if converted_video and os.path.exists(converted_video):
                total_time = time.time() - start_time
                logger.info(f"Video converted successfully: {converted_video}")
                yield converted_video, f"Video converted successfully: {converted_video}\nTotal generation time: {total_time:.1f}s"
                return converted_video, f"Video generation completed successfully!\nTotal generation time: {total_time:.1f}s"
            else:
                logger.error("Video conversion failed")
                return None, "Error: Video conversion failed"
        logger.error("Failed to generate video")
        return None, "Error: Failed to generate video"
        
    except Exception as e:
        logger.error(f"Error in process_with_status: {str(e)}")
        return None, f"Error: {str(e)}"

def validate_flow_shift(flow_shift, resolution):
    """Validate flow_shift based on resolution"""
    if "480" in resolution:
        return min(max(flow_shift, 0.1), 3.0)
    return min(max(flow_shift, 0.1), 10.0)

def validate_inference_steps(steps):
    """Validate inference steps"""
    return min(max(steps, 1), 100)

def validate_fps(fps):
    """Validate FPS"""
    return min(max(fps, 1), 30)

def validate_speed(speed):
    """Validate speed multiplier"""
    return min(max(speed, 0.1), 2.0)

def validate_guiding_scale(scale):
    """Validate guiding scale"""
    return min(max(scale, 0.1), 10.0)

def validate_lora_scale(scale):
    """Validate LoRA scale"""
    return min(max(scale, 0.1), 2.0)

# Custom CSS for Replicate-like styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
#component-0 {
    max-width: 1200px !important;
    margin: 0 auto !important;
}
.status-box {
    font-family: monospace;
    white-space: pre-wrap;
}
.video-container {
    position: relative;
    min-height: 400px;
    background: #f5f5f5;
    border-radius: 8px;
}
"""

# Launch Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("""
    # Wan 2.1 Video Generator
    """)

    with gr.Row():
        with gr.Column(scale=1):
            task = gr.Dropdown(
                choices=list(TASK_MAP.keys()),
                value="Image2Video",
                label="Generation Task",
                info="Choose between Text-to-Video or Image-to-Video generation"
            )
            
            image = gr.Image(
                type="filepath",
                label="Upload Image",
                visible=True,
                info="Upload an image for Image-to-Video generation"
            )
            
            prompt = gr.Textbox(
                label="Text Prompt",
                placeholder="Describe your video... (e.g., A cinematic scene with dynamic camera movement)",
                lines=3,
                info="Enter a detailed description of the video you want to generate"
            )
            
            resolution = gr.Dropdown(
                choices=list(RESOLUTION_MAP.keys()),
                value="Horizontal (832x480)",
                label="Video Resolution",
                info="Select the desired output resolution (480p recommended for better performance)"
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=40,
                    step=1,
                    label="Number of Inference Steps",
                    info="More steps generally produce better quality but take longer"
                )
                
                seed = gr.Number(
                    value=-1,
                    label="Seed",
                    info="Set to -1 for random seed"
                )
                
                flow_shift = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=3.0,
                    step=0.1,
                    label="Flow Shift",
                    info="Controls the flow matching scheduler shift factor (3.0 recommended for 480p)"
                )

                fps = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=8,
                    step=1,
                    label="FPS",
                    info="Frames per second for the output video"
                )

                speed = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Speed",
                    info="Speed multiplier for the output video"
                )

                sample_guiding_scale = gr.Slider(
                    minimum=0.1,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                    label="Sample Guiding Scale",
                    info="Guiding scale for sampling"
                )

                lora_weights = gr.Textbox(
                    label="LoRA Weights Path",
                    placeholder="Enter Hugging Face path (e.g., hf://username/repo/filename.safetensors) or local path",
                    info="Enter Hugging Face path (hf://username/repo/filename.safetensors) or local path to LoRA weights"
                )

                lora_scale = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="LoRA Scale",
                    info="Scale factor for LoRA weights"
                )
            
            run_btn = gr.Button("Generate Video", variant="primary", size="large")
            
        with gr.Column(scale=1):
            with gr.Box(elem_classes=["video-container"]) as video_box:
                output_video = gr.Video(
                    label="Generated Video",
                    visible=True
                )
            
            status_text = gr.Textbox(
                label="Generation Status",
                value="Ready to generate video...",
                interactive=False,
                lines=10,
                elem_classes=["status-box"]
            )

    def update_interface(task):
        return gr.Image(visible=(task == "Image2Video"))

    def validate_all_params(flow_shift, resolution, num_inference_steps, fps, speed, sample_guiding_scale, lora_scale, task):
        """Validate all parameters before generation"""
        flow_shift = validate_flow_shift(flow_shift, resolution)
        num_inference_steps = validate_inference_steps(num_inference_steps)
        fps = validate_fps(fps)
        speed = validate_speed(speed)
        sample_guiding_scale = validate_guiding_scale(sample_guiding_scale)
        lora_scale = validate_lora_scale(lora_scale)
        return flow_shift, num_inference_steps, fps, speed, sample_guiding_scale, lora_scale

    task.change(
        fn=update_interface,
        inputs=[task],
        outputs=[image]
    )

    flow_shift.change(
        fn=validate_flow_shift,
        inputs=[flow_shift, resolution],
        outputs=[flow_shift]
    )

    resolution.change(
        fn=validate_flow_shift,
        inputs=[flow_shift, resolution],
        outputs=[flow_shift]
    )

    num_inference_steps.change(
        fn=validate_inference_steps,
        inputs=[num_inference_steps],
        outputs=[num_inference_steps]
    )

    fps.change(
        fn=validate_fps,
        inputs=[fps],
        outputs=[fps]
    )

    speed.change(
        fn=validate_speed,
        inputs=[speed],
        outputs=[speed]
    )

    sample_guiding_scale.change(
        fn=validate_guiding_scale,
        inputs=[sample_guiding_scale],
        outputs=[sample_guiding_scale]
    )

    lora_scale.change(
        fn=validate_lora_scale,
        inputs=[lora_scale],
        outputs=[lora_scale]
    )

    run_btn.click(
        fn=validate_all_params,
        inputs=[flow_shift, resolution, num_inference_steps, fps, speed, sample_guiding_scale, lora_scale, task],
        outputs=[flow_shift, num_inference_steps, fps, speed, sample_guiding_scale, lora_scale]
    ).then(
        fn=process_with_status,
        inputs=[image, prompt, resolution, task, num_inference_steps, seed, flow_shift, 
               fps, speed, sample_guiding_scale, lora_weights, lora_scale],
        outputs=[output_video, status_text],
        api_name="generate_video",
        queue=True
    )

demo.queue()
demo.launch(share=True)
