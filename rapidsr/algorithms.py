import abc
import os
import statistics
from typing import Iterable
import numpy as np
import torch
import cv2
import dgl
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

from rapidsr import stop, log, conversion
from rapidsr import _models
from rapidsr import _internal


def _neural_network_resolve(image: np.ndarray, model: nn.Module,
                            device: torch.device) -> np.ndarray:
    original_ndim = image.ndim
    if image.ndim == 2:
        # ToTensor works only with 3 dimensional arrays
        image = image.reshape(list(image.shape) + [1])
    model.eval()
    with torch.no_grad():
        tensor = transforms.ToTensor()(image).to(dtype=torch.float32)
        batch = torch.unsqueeze(tensor, 0)
        output = model(batch.to(device))
    output_image = output.cpu().data[0].numpy()
    output_image = np.moveaxis(output_image, 0, -1)
    output_image = conversion.convert_float_to_byte_image(output_image)
    if original_ndim == 2:
        output_image = output_image.reshape(output_image.shape[:2])
    return output_image


def _graph_network_resolve(graph: dgl.DGLGraph, model: nn.Module,
                           device: torch.device) -> dgl.DGLGraph:
    model.eval()
    with torch.no_grad():
        graph.to(device)
        return model(graph)


class Trainable(log.Observable):

    @abc.abstractmethod
    def get_model(self) -> nn.Module:
        pass

    def train(self, epochs: int,
              training_data_loader: data.DataLoader,
              validation_data_loader: data.DataLoader,
              loss_function: nn.Module,
              optimizer: optim.Optimizer,
              stop_condition: stop.StopCondition,
              device: torch.device):
        for epoch in range(1, epochs + 1):
            self.notify_of_event(log.LogEvent.TrainEpochStarted, epoch)
            training_loss = self._run_training_part(training_data_loader,
                                                    optimizer,
                                                    loss_function,
                                                    device)
            self.notify_of_event(log.LogEvent.TrainEpochFinished,
                                 log.LossData(training_loss, epoch))
            self.notify_of_event(log.LogEvent.ValidationEpochStarted, epoch)
            validation_loss = self._run_validation_part(validation_data_loader,
                                                        loss_function, device)
            self.notify_of_event(log.LogEvent.ValidationEpochFinished,
                                 log.LossData(validation_loss, epoch))
            self.notify_of_event(log.LogEvent.EpochFinished,
                                 log.WeightsData(
                                     self.get_model().state_dict(), epoch))
            if stop_condition.is_satisfied(
                    stop.StopInfo(training_loss, validation_loss, epoch)):
                break

    def save_weights(self, path: os.PathLike):
        torch.save(self.get_model().state_dict(), path)

    def load_weights(self, path: os.PathLike):
        weights = torch.load(path)
        self.get_model().load_state_dict(weights)

    def _run_training_part(self, data_loader, optimizer, loss_function,
                           device: torch.device) -> float:
        model: nn.Module = self.get_model()
        model.train(True)
        batch_losses = []
        with torch.set_grad_enabled(True):
            for input, target in data_loader:
                if isinstance(input, dgl.DGLGraph):
                    input.to(device)
                    target.to(device)
                else:
                    input = input.to(device)
                    target = target.to(device)
                optimizer.zero_grad()
                output = model(input)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().numpy()))
        return statistics.mean(batch_losses) if batch_losses else 0.0

    def _run_validation_part(self, data_loader, loss_function,
                             device: torch.device) -> float:
        model: nn.Module = self.get_model()
        model.train(False)
        batch_losses = []
        with torch.no_grad():
            for input, target in data_loader:
                if isinstance(input, dgl.DGLGraph):
                    input.to(device)
                    target.to(device)
                else:
                    input = input.to(device)
                    target = target.to(device)
                output = model(input)
                loss = loss_function(output, target)
                batch_losses.append(float(loss.detach().numpy()))
        return statistics.mean(batch_losses) if batch_losses else 0.0


class SuperResolutionAlgorithm(abc.ABC):
    pass


class SingleImageAlgorithm(SuperResolutionAlgorithm):

    @abc.abstractmethod
    def resolve(self, image: np.ndarray) -> np.ndarray:
        pass


class MultipleImageAlgorithm(SuperResolutionAlgorithm):

    @abc.abstractmethod
    def resolve(self, images: Iterable[np.ndarray]) -> np.ndarray:
        pass


class SRCNN(SingleImageAlgorithm, Trainable):
    def __init__(self, channels: int,
                 upscale_factor: int,
                 device: torch.device,
                 extraction_kernel_size: int = 9,
                 mapping_kernel_size: int = 1,
                 reconstruction_kernel_size: int = 5,
                 extraction_layer_kernels: int = 64,
                 mapping_layer_kernels: int = 32):
        """
        :param channels: Amount of channels in the image e.g. 3 for RGB, 1 for
        grayscale.
        :param upscale_factor: How many times to increase the size of an image.
        :param device: Device to be utilized during image resolving.
        :param extraction_kernel_size: Size of the feature extraction kernel.
        :param mapping_kernel_size: Size of the mapping kernel.
        :param reconstruction_kernel_size: Size of the reconstruction kernel.
        :param extraction_layer_kernels: Amount of kernels in the extraction
        layer.
        :param mapping_layer_kernels: Amount of kernels in the mapping layer.
        """
        super().__init__()
        self._model = _models.SRCNN(channels, extraction_kernel_size,
                                    mapping_kernel_size,
                                    reconstruction_kernel_size,
                                    extraction_layer_kernels,
                                    mapping_layer_kernels)
        self._upscale_factor = upscale_factor
        self._device = device

    def get_model(self) -> nn.Module:
        return self._model

    def resolve(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, dsize=None, fx=self._upscale_factor,
                           fy=self._upscale_factor,
                           interpolation=cv2.INTER_CUBIC)
        return _neural_network_resolve(image, self._model, self._device)


class FSRCNN(SingleImageAlgorithm, Trainable):
    def __init__(self,
                 channels: int,
                 upscale_factor: int,
                 device: torch.device,
                 extraction_kernel_size: int = 5,
                 shrinkage_kernel_size: int = 1,
                 map_kernel_size: int = 3,
                 expansion_kernel_size: int = 1,
                 deconvolution_kernel_size: int = 9,
                 n_map_layers: int = 4,
                 n_dimension_filters: int = 56,
                 n_shrinkage_filters: int = 12):
        """

        :param channels: Amount of channels in an image.
        RGB = 3 channels, grayscale = 1 channel etc.
        :param upscale_factor: How many times to increase the size of an image.
        :param device: Device to be utilized during image resolving.
        :param extraction_kernel_size: Size of the kernel in the extraction
        layer.
        :param shrinkage_kernel_size:Size of the kernel in the shrinkage layer.
        :param map_kernel_size:Size of the kernel in the map layers.
        :param expansion_kernel_size:Size of the kernel in the expansion
        layer.
        :param deconvolution_kernel_size: Size of the kernel in the
        deconvolution layer.
        :param n_map_layers: Amount of map layers.
        :param n_dimension_filters: Amount of filters before shrinking and
        after expansion.
        :param n_shrinkage_filters: Amount of kernels in layers after
        shrinking.
        """
        super().__init__()
        self._model = _models.FSRCNN(channels, upscale_factor,
                                     extraction_kernel_size,
                                     shrinkage_kernel_size, map_kernel_size,
                                     expansion_kernel_size,
                                     deconvolution_kernel_size, n_map_layers,
                                     n_dimension_filters, n_shrinkage_filters)
        self._device = device

    def get_model(self) -> nn.Module:
        return self._model

    def resolve(self, image: np.ndarray) -> np.ndarray:
        return _neural_network_resolve(image, self._model, self._device)


class LapSRN(SingleImageAlgorithm, Trainable):
    """

    """

    def __init__(self, channels: int, upscale_factor: int,
                 device: torch.device, depth: int = 5,
                 recursive_blocks_count: int = 8):
        super().__init__()
        self._model = _models.LapSRN(channels, upscale_factor, depth,
                                     recursive_blocks_count)
        self._device = device

    def get_model(self) -> nn.Module:
        return self._model

    def resolve(self, image: np.ndarray) -> np.ndarray:
        return _neural_network_resolve(image, self._model, self._device)


class SRGCN(SingleImageAlgorithm, Trainable):
    """

    """

    def __init__(self, channels: int, upscale_factor: int,
                 device: torch.device):
        super().__init__()
        self._upscale_factor = upscale_factor
        self._model = _models.SRGCN(channels)
        self._device = device

    def get_model(self) -> nn.Module:
        return self._model

    def resolve(self, image: np.ndarray) -> np.ndarray:
        image = cv2.resize(image, dsize=None, fx=self._upscale_factor,
                           fy=self._upscale_factor,
                           interpolation=cv2.INTER_CUBIC)
        graph = conversion.image_to_graph(image)
        sr_graph = _graph_network_resolve(graph, self._model, self._device)
        return conversion.graph_to_image(sr_graph)
