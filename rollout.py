from enum import Enum
import random
import art

import openai
import time
import math
import requests
from pydantic import BaseModel
from connect4 import Connect4, Player
from openpipe.client import UpdateLogTagsRequestFiltersItem, AsyncOpenPipe
from art.local import LocalBackend
from dataclasses import dataclass
from openai import AsyncOpenAI

@dataclass
class Config:
    max_completion_tokens: int = 128


class ScenarioConnect4(BaseModel):
    step: int


class Opponent(Enum):
    RANDOM = "random"
    EVAL = "eval"

async def make_opponent_move(game: Connect4, opponent: Opponent) -> int:
    if opponent == Opponent.RANDOM:
        return random.choice(game.get_valid_moves())
    elif opponent == Opponent.EVAL:
        # client = AsyncOpenA()
        messages=[
            {
                "role": "system",
                "content": "You are an excellent Connect 4 player. Always choose the next move that most likely to lead to a win. Return your move as an XML object with a single property 'move', like so: <move>{column index}</move>. The columns are zero-indexed. The board is as follows: {game.render()}",
            }
        ]
        return 0

    else:
        raise ValueError(f"Invalid opponent: {opponent}")


@art.retry(exceptions=(openai.LengthFinishReasonError, requests.ReadTimeout))
async def rollout(
    model: art.Model, scenario: ScenarioConnect4, op_client: AsyncOpenPipe, config: Config, opponent: Opponent
) -> art.Trajectory:
    game = Connect4()

    move_number = 0

    # TODO: Currently the model is hard-coded to start first.
    # We need to change this later so that it sometimes start second.

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are an excellent Connect 4 player. Always choose the next move that most likely to lead to a win. Return your move as an XML object with a single property 'move', like so: <move>{column index}</move>. The columns are zero-indexed.",
            }
        ],
        reward=0,
    )

    while True:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": game.render()}
        )

        requested_at = int(time.time() * 1000)
        messages = trajectory.messages()

        async def get_completion():
            client = model.openai_client()
            return await client.chat.completions.create(
                max_completion_tokens=config.max_completion_tokens,
                messages=messages,
                model=model.name,
            )

        try:
            chat_completion = await get_completion()
            last_completion = chat_completion
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e

        try:
            if op_client.api_key:
                await op_client.report(
                    requested_at=requested_at,
                    received_at=int(time.time() * 1000),
                    req_payload={
                        "model": model.name,
                        "messages": messages,
                        "metadata": {
                            "game_id": game.id,
                            "notebook-id": "rollout",
                            "step": str(scenario.step),
                            "move_number": str(move_number),
                        },
                    },
                    resp_payload=chat_completion,
                    status_code=200,
                )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        # content: <move>0</move>
        def try_make_move(content: str):
            try:
                move = int(content.split("<move>")[1].split("</move>")[0])
                move_successful, _ = game.make_move(move)
                if not move_successful:
                    raise ValueError("Invalid move")
            except Exception as e:
                raise ValueError(f"Invalid move: {e}")

        try:
            # make a move based on the LLM's output
            try_make_move(content)
            # apply random valid opponent move.
            opponent_move = await make_opponent_move(game, opponent)
            game.make_move(opponent_move)
            move_number += 1
        except ValueError:
            trajectory.reward = -1
            break

        if game.game_over:
            # Win: 1, Draw: 0.5, Lose: 0, Bad formatting: -1
            if game.winner == Player.PLAYER1:
                trajectory.reward = 1
            elif game.winner == Player.PLAYER2:
                trajectory.reward = 0
            else:
                trajectory.reward = 0.5
            break

    try:
        if op_client.api_key:
            await op_client.update_log_metadata(
                filters=[
                    UpdateLogTagsRequestFiltersItem(
                        field="completionId",
                        equals=last_completion.id,
                    )
                ],
                metadata={
                    "reward": str(trajectory.reward),
                    "reward_assigned": "true",
                },
            )
    except Exception as e:
        print(f"Error updating log metadata: {e}")

    return trajectory