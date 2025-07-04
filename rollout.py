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

from solver import Connect4Solver
from config import Config

def extract_move(content: str) -> int:
    return int(content.split("<move>")[1].split("</move>")[0])


class ScenarioConnect4(BaseModel):
    step: int


class Opponent(str, Enum):
    RANDOM = "random"
    EVAL = "eval"
    SOLVER = "solver"

async def make_opponent_move(game: Connect4, opponent: Opponent, difficulty: float = 0) -> int | None:
    if opponent == Opponent.RANDOM:
        return random.choice(game.get_valid_moves())
    elif opponent == Opponent.SOLVER:

        # Difficulty = 1 means always use the solver
        if random.random() < difficulty:
            solver = Connect4Solver()
            move = solver.get_best_move(game)
            if move is None:
                return random.choice(game.get_valid_moves())
        else:
            return random.choice(game.get_valid_moves())

    elif opponent == Opponent.EVAL:
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an excellent Connect 4 player. Always choose the next move that most likely to lead to a win. Return your move as an XML object with a single property 'move', like so: <move>{column index}</move>. The columns are zero-indexed. You are X.",
                },
                {
                    "role": "user",
                    "content": f"{game.render()}",
                }
            ],
            model="gpt-4o",
            max_completion_tokens=512,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content returned from OpenAI completion.")
        try:
            return extract_move(content)
        except Exception as e:
            raise ValueError(f"Invalid move: {e}")


@art.retry(exceptions=(openai.LengthFinishReasonError, requests.ReadTimeout))
async def rollout(
    model: art.Model, scenario: ScenarioConnect4, op_client: AsyncOpenPipe, config: Config, opponent: Opponent, difficulty: float = 0.5
) -> art.Trajectory:
    game = Connect4()

    move_number = 0


    # TODO: Currently the model is hard-coded to start first.
    # We need to change this later so that it sometimes start second.

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are an excellent Connect 4 player. Always choose the next move that most likely to lead to a win. Return your move as an XML object with a single property 'move', like so: <move>{column index}</move>. The columns are zero-indexed. You are player X.",
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
                temperature=1.0,
            )

        try:
            chat_completion = await get_completion()
            last_completion = chat_completion
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

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
            k = 0
            while k < 5:
                # random or solver.
                if random.random() < difficulty:
                    opponent_move = await make_opponent_move(game, opponent, difficulty=0)
                else:
                    opponent_move = random.choice(game.get_valid_moves())

                if opponent_move is not None:
                    break
                k += 1

            if opponent_move is not None:
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