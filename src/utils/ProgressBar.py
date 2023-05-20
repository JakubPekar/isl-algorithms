from timeit import default_timer as timer
from src.utils.utils import get_time


class ProgressBar:
    def __init__(
        self, total: int, batch_size: int, bar_length: int = 50
    ) -> None:
        self.total = total
        self.batch_size = batch_size
        self.bar_length = bar_length
        self.__time = timer()

    def __call__(self, current: int, **kwargs: float) -> None:
        if current == self.total:
            arrow = '=' * (current * self.bar_length // self.total)
            end = '\n'
        else:
            length = (current * self.bar_length - 1) // self.total
            arrow = '=' * length + '>'
            end = '\r'

        spaces = ' ' * (self.bar_length - len(arrow))
        bar = f'[{arrow}{spaces}]'

        batch = f'{current}/{self.total}'
        time = f'{get_time((timer() - self.__time) / self.batch_size)}/sample'
        others = ' - '.join(': '.join(map(str, x)) for x in kwargs.items())

        print(f'{batch}  {bar}  -  {time}  -  {others}', end=end, flush=True)
        self.__time = timer()
