import art
import os
from dotenv import load_dotenv
import random

from openpipe.client import OpenPipe
from art.local import LocalBackend

from rollout import Config, Opponent, ScenarioConnect4, rollout

load_dotenv()



async def train():
    op_client = OpenPipe()
    print("OpenPipe client initialized")

    random.seed(42)

    # Use local backend with persistent volume
    backend = LocalBackend(path="/root/workspace/.art")

    model = art.TrainableModel(
        name="001-script", project="connect4-local", base_model="Qwen/Qwen2.5-3B-Instruct"
    )
    await model.register(backend)
    config = Config()

    for i in range(await model.get_step(), 50):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, ScenarioConnect4(step=i), op_client, config, Opponent.RANDOM) for _ in range(48)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=5e-5))