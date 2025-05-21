from huggingface_hub import snapshot_download

# Download the T2V-14B model
model_path = snapshot_download(
    repo_id="Wan-Video/Wan2.1-T2V-14B",
    local_dir="./Wan2.1-T2V-14B",
    local_dir_use_symlinks=False
)

print(f"Model downloaded to: {model_path}") 