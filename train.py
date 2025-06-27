import art
import os
from dotenv import load_dotenv
import random

from openpipe.client import AsyncOpenPipe
from art.local import LocalBackend

from rollout import Config, Opponent, ScenarioConnect4, rollout

load_dotenv()

async def train():
    op_client = AsyncOpenPipe()
    config = Config()
    print("OpenPipe client initialized")

    if config.opponent.lower() == "random":
        opponent = Opponent.RANDOM
    elif config.opponent.lower() == "eval":
        opponent = Opponent.EVAL
    else:
        raise ValueError(f"Invalid opponent: {config.opponent}")

    random.seed(42)

    # Use local backend with persistent volume
    backend = LocalBackend(path="/root/workspace/.art")

    model = art.TrainableModel(
        name=config.experiment_name, project="connect4-local", base_model=config.model
    )
    await model.register(backend)

    for i in range(await model.get_step(), config.max_steps):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, ScenarioConnect4(step=i), op_client, config, Opponent.RANDOM) for _ in range(config.group_size)
                )
                for _ in range(config.groups_per_step)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=config.learning_rate, beta=config.beta))