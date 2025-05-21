import json
import os

def create_t2v_workflow():
    """Create a Text-to-Video workflow for ComfyUI"""
    workflow = {
        "3": {
            "class_type": "WanT2V",
            "inputs": {
                "prompt": "A beautiful sunset over a calm ocean, waves gently rolling, birds flying in the distance",
                "negative_prompt": "blurry, low quality, distorted",
                "resolution": "1280*720",  # 720P
                "num_inference_steps": 40,
                "guidance_scale": 7.5,
                "seed": -1,  # Random seed
                "sample_shift": 5.0,
                "model": "Wan2.1-T2V-14B"
            }
        },
        "4": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename": "t2v_output",
                "format": "mp4"
            }
        }
    }
    
    # Connect nodes
    workflow["4"]["inputs"]["video"] = ["3", 0]
    
    return workflow

def create_i2v_workflow():
    """Create an Image-to-Video workflow for ComfyUI"""
    workflow = {
        "1": {
            "class_type": "LoadImage",
            "inputs": {
                "image": "input_image.jpg"
            }
        },
        "2": {
            "class_type": "WanI2V",
            "inputs": {
                "image": ["1", 0],
                "prompt": "A cinematic scene with dynamic camera movement",
                "negative_prompt": "blurry, low quality, distorted",
                "resolution": "1280*720",  # 720P
                "num_inference_steps": 40,
                "guidance_scale": 7.5,
                "seed": -1,  # Random seed
                "sample_shift": 5.0,
                "model": "Wan2.1-I2V-14B-720P"
            }
        },
        "3": {
            "class_type": "SaveVideo",
            "inputs": {
                "filename": "i2v_output",
                "format": "mp4"
            }
        }
    }
    
    # Connect nodes
    workflow["3"]["inputs"]["video"] = ["2", 0]
    
    return workflow

def save_workflow(workflow, filename):
    """Save workflow to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(workflow, f, indent=2)

def main():
    # Create workflows directory if it doesn't exist
    os.makedirs("workflows", exist_ok=True)
    
    # Create and save T2V workflow
    t2v_workflow = create_t2v_workflow()
    save_workflow(t2v_workflow, "workflows/t2v_workflow.json")
    print("T2V workflow saved to workflows/t2v_workflow.json")
    
    # Create and save I2V workflow
    i2v_workflow = create_i2v_workflow()
    save_workflow(i2v_workflow, "workflows/i2v_workflow.json")
    print("I2V workflow saved to workflows/i2v_workflow.json")

if __name__ == "__main__":
    main() 