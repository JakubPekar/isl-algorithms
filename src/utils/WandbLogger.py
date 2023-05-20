import wandb
from typing import List
from src.utils.types import TrainingParams


class WandbLogger:
    def __init__(
        self, project: str, config: TrainingParams, tags: List[str] = []
    ) -> None:
        wandb.init(project=project, tags=tags, config=config) # type: ignore
        config.update(wandb.config)

    def __call__(self, **kwargs: float) -> None:
        wandb.log(kwargs)
