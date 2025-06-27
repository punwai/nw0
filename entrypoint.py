"""
modal run -d entrypoint.py
"""

import os
from datetime import datetime
from pathlib import Path
import modal

MODAL_TOKEN = Path("~/.modal.toml").expanduser()
def build_modal_image() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("procps")
        .pip_install(
            "numpy==1.26.4",
            "openpipe>=4.49.0",
            "openpipe-art==0.3.7",
            "chz>=0.3.0",
            "s3fs",
            "pyinstrument",
            "python-dotenv",
            "openai",
            "requests",
            "pydantic",
        )
        .copy_local_file("train.py", "/root/train.py")
        .copy_local_file("rollout.py", "/root/rollout.py")
        .copy_local_file("connect4.py", "/root/connect4.py")
        .add_local_file(MODAL_TOKEN, remote_path="/root/.modal.toml")
    )


app = modal.App(
    name=f"rlapp-{os.environ.get('USER', 'unknown')}-{datetime.now().strftime('%b%d-%I-%M%p').lower()}"
)
APP_TIMEOUT = 24 * 60 * 60

image = build_modal_image()


@app.function(
    image=image,
    gpu="H100:1",
    timeout=APP_TIMEOUT,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
)
async def train_remote():
    from train import train
    return await train()


@app.local_entrypoint()
def run():
    train_remote.remote()
