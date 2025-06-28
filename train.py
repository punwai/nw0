import art
import os
from dotenv import load_dotenv
import random

from openpipe.client import AsyncOpenPipe
from art.local import LocalBackend

from rollout import Opponent, ScenarioConnect4, rollout
from config import Config

load_dotenv()

async def eval():
    config = Config()
    model = art.TrainableModel(
        name=config.experiment_name, project="connect4-local", base_model=config.model
    )
    backend = LocalBackend(path="/root/workspace/.art")

    await model.register(backend)
    op_client = AsyncOpenPipe()
    opponent = Opponent.EVAL
    difficulty = 0.0

    eval_groups = []
    for i in range(config.eval_batch_size):
        eval_groups.append(
            art.TrajectoryGroup(
                [rollout(model, ScenarioConnect4(step=i), op_client, config, opponent, difficulty=difficulty)]
            )
        )

    eval_groups = await art.gather_trajectory_groups(eval_groups, pbar_desc="gather")
    print([trajectory.reward for group in eval_groups for trajectory in group.trajectories])

async def train():
    op_client = AsyncOpenPipe()
    config = Config()
    print("OpenPipe client initialized")

    if config.opponent.lower() == "random":
        opponent = Opponent.RANDOM
    elif config.opponent.lower() == "eval":
        opponent = Opponent.EVAL
    elif config.opponent.lower() == "solver":
        opponent = Opponent.SOLVER
    else:
        raise ValueError(f"Invalid opponent: {config.opponent}")

    random.seed(42)

    # Use local backend with persistent volume
    backend = LocalBackend(path="/root/workspace/.art")

    model = art.TrainableModel(
        name=config.experiment_name, project="connect4-local", base_model=config.model
    )
    await model.register(backend)

    possible_difficulties = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in range(await model.get_step(), config.max_steps):
        train_groups = []

        for _ in range(config.groups_per_step):
            difficulty = random.choice(possible_difficulties)
            train_groups.append(
                art.TrajectoryGroup(
                    rollout(model, ScenarioConnect4(step=i), op_client, config, opponent, difficulty=difficulty) for _ in range(config.group_size)
                )
            )

        train_groups = await art.gather_trajectory_groups(train_groups, pbar_desc="gather")
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=config.learning_rate, beta=config.beta))