import torch


class Checkpoint:
    def __init__(
        self, module: torch.nn.Module, path: str, overfitting: int = 20
    ) -> None:
        self.module = module
        self.overfitting = overfitting
        self.path = path
        self.__score = 10
        self.__overfitting = 0

    def __call__(self, score: float) -> None:
        if score < self.__score:
            print('🚀 Checkpoint 🚀')
            torch.save(self.module.state_dict(), self.path)
            self.__score = score
            self.__overfitting = 0
        else:
            self.__overfitting += 1

    def is_overfitting(self) -> bool:
        return self.__overfitting > self.overfitting
