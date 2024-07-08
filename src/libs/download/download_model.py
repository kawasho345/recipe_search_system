from huggingface_hub import snapshot_download

def download_model(name):
    snapshot_download(
        repo_id=name,
        local_dir = f"model/{name}",
        local_dir_use_symlinks=False
    )