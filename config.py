from dataclasses import dataclass

@dataclass
class Config:
    experiment_name: str = "debug_gs_8_ng_8_lr5e-6_beta0.05_against_solver"
    opponent: str = "eval"
    max_completion_tokens: int = 128
    learning_rate: float = 1e-6
    beta: float = 0.10
    group_size: int = 16
    groups_per_step: int = 8
    max_steps: int = 100
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    eval_model_name: str = "gpt-4o"
    eval_max_completion_tokens: int = 512
    eval_batch_size: int = 64

# 46
# 
