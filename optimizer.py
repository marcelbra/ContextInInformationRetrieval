from torch import Tensor

class Optimizer:

    def __init__(self,
                 current=None,
                 strategy="Default"):
        self._current = current
        self._strategy = strategy

    def step(self, location: Tensor) -> Tensor:
        """Makes a step in conceptual space according to a given
        strategy. Expexts the current location of the searcher,
        performs the wished strategy, changes the location in-place
        and returns the final location."""
        self.current = location
        self.strategy()
        return self.current

    @property
    def current(self):
        return self._current

    @current.setter
    def current(self, new_value: Tensor) -> None:
        self._current = new_value

    def strategy(self, mode: str):
        if mode == "Default":
            self.strategy_one()

    def strategy_one(self):
        pass