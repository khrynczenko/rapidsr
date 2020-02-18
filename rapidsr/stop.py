import abc
from typing import NamedTuple


class StopInfo(NamedTuple):
    training_loss: float
    validation_loss: float
    epoch: int


class StopCondition(abc.ABC):
    @abc.abstractmethod
    def is_satisfied(self, info: StopInfo) -> bool:
        """
        Returns true if condition is satisfied.
        :param info: Information used by different stop conditions.
        """

        pass


class NoCondition(StopCondition):
    def is_satisfied(self, info: StopInfo) -> bool:
        return False


class StopWhenValidationLossHasNotImproved(StopCondition):
    def __init__(self, n_epochs: int):
        """

        :param n_epochs: It in the last n_epochs validation_loss has not
        improved the condition will be satisfied.
        """
        self._n_epochs = n_epochs
        self._losses = []

    def is_satisfied(self, info: StopInfo) -> bool:
        self._losses.append(info.validation_loss)
        if len(self._losses) >= 50:
            loss_ = self._losses[-50]
            differences = [(loss_ - loss) > 0.0 for loss in
                           self._losses[-1:-50:-1]]
            return any(differences)
        return False
