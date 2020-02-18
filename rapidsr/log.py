import abc
import enum
import os
import csv
import timeit
from typing import NamedTuple, Any, List

import torch


class LossData(NamedTuple):
    loss: float
    epoch: int


class WeightsData(NamedTuple):
    weights: Any
    epoch: int


class LogEvent(enum.Enum):
    TrainEpochStarted = enum.auto()
    TrainEpochFinished = enum.auto()
    ValidationEpochStarted = enum.auto()
    ValidationEpochFinished = enum.auto()
    EpochFinished = enum.auto()


class Logger(abc.ABC):
    @abc.abstractmethod
    def on_event(self, event: LogEvent, event_data: Any):
        pass


class WeightsLogger(Logger):
    def __init__(self, log_directory: os.PathLike, filename: str):
        self._log_directory = log_directory
        self._filename = filename

    def on_event(self, event: LogEvent, event_data: WeightsData):
        if event == LogEvent.EpochFinished:
            weights_path = os.path.join(self._log_directory,
                                        self._filename +
                                        f"_{str(event_data.epoch).zfill(6)}")
            with open(weights_path, "wb") as file:
                torch.save(event_data.weights, file)


class TimeLogger(Logger):
    def __init__(self, log_directory: os.PathLike, filename: str):
        self._log_directory = log_directory
        self._filename = filename
        self._filename_abs = os.path.join(self._log_directory,
                                          self._filename)
        self.time_start_points = []
        self.time_start_points = []
        self._time_point = None

        with open(self._filename_abs, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "epoch_time"])

    def on_event(self, event: LogEvent, event_data: LossData):
        if event == LogEvent.TrainEpochStarted:
            self._time_point = timeit.default_timer()
        elif event == LogEvent.TrainEpochFinished:
            with open(self._filename_abs, "a") as file:
                writer = csv.writer(file)
                time = timeit.default_timer() - self._time_point
                writer.writerow([event_data.epoch, f"{str(time).zfill(2)}"])


class LossLogger(Logger):
    def __init__(self, log_directory: os.PathLike, filename: str):
        self._log_directory = log_directory
        self._filename = filename
        self._filename_abs = os.path.join(self._log_directory,
                                          self._filename)

        with open(self._filename_abs, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "epoch_time"])

    def on_event(self, event: LogEvent, event_data: LossData):
        if event == LogEvent.TrainEpochFinished:
            with open(self._filename_abs, "a") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [event_data.epoch, f"{str(event_data.loss).zfill(6)}"])


class Observable:
    def __init__(self):
        self._loggers: List[Logger] = []

    def attach_logger(self, logger: Logger):
        self._loggers.append(logger)

    def notify_of_event(self, event: LogEvent, event_data: Any):
        for logger in self._loggers:
            logger.on_event(event, event_data)
