import pytest
import numpy as np
import torch
import dgl
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from rapidsr import algorithms, loss, stop, conversion


class TestSRCNN:
    @pytest.mark.parametrize(["upscale_factor", "input_size", "output_size"],
                             [(2, (20, 20), (40, 40))
                                 , (2, (23, 23), (46, 46))
                                 , (3, (20, 20), (60, 60))
                                 , (4, (23, 23), (92, 92))
                              ])
    def test_resolve(self, upscale_factor, input_size, output_size):
        device = torch.device("cpu")
        srcnn = algorithms.SRCNN(1, upscale_factor, device)
        image = np.ones(input_size, dtype=np.uint8)
        output = srcnn.resolve(image)
        assert output.shape == output_size

    def test_train(self):
        device = torch.device("cpu")
        srcnn = algorithms.SRCNN(1, 2, device)
        lr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
        hr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
        lr_tensors = [transforms.ToTensor()(image) for image in lr_images]
        hr_tensors = [transforms.ToTensor()(image) for image in hr_images]
        loader = data.DataLoader(list(zip(lr_tensors, hr_tensors)))
        srcnn.train(2, loader, loader, nn.MSELoss(),
                    optim.Adam(srcnn.get_model().parameters()),
                    stop.NoCondition(), device)


class TestFSRCNN:
    @pytest.mark.parametrize(["upscale_factor", "input_size", "output_size"],
                             [(2, (20, 20), (40, 40))
                                 , (2, (23, 23), (46, 46))
                                 , (3, (20, 20), (60, 60))
                                 , (4, (23, 23), (92, 92))
                              ])
    def test_resolve(self, upscale_factor, input_size, output_size):
        device = torch.device("cpu")
        fsrcnn = algorithms.FSRCNN(1, upscale_factor, device)
        image = np.ones(input_size, dtype=np.uint8)
        output = fsrcnn.resolve(image)
        assert output.shape == output_size

    def test_train(self):
        device = torch.device("cpu")
        fsrcnn = algorithms.FSRCNN(1, 2, device)
        lr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
        hr_images = [np.ones((40, 40), dtype=np.uint8) for _ in range(3)]
        lr_tensors = [transforms.ToTensor()(image) for image in lr_images]
        hr_tensors = [transforms.ToTensor()(image) for image in hr_images]
        loader = data.DataLoader(list(zip(lr_tensors, hr_tensors)))
        fsrcnn.train(2, loader, loader, nn.MSELoss(),
                     optim.Adam(fsrcnn.get_model().parameters()),
                     stop.NoCondition(), device)


class TestLapSRN:
    @pytest.mark.parametrize(["upscale_factor", "input_size", "output_size"],
                             [(2, (20, 20), (40, 40))
                                 , (2, (23, 23), (46, 46))
                                 , (4, (23, 23), (92, 92))
                              ])
    def test_resolve(self, upscale_factor, input_size, output_size):
        device = torch.device("cpu")
        lapsrn = algorithms.LapSRN(1, upscale_factor, device)
        image = np.ones(input_size, dtype=np.uint8)
        output = lapsrn.resolve(image)
        assert output.shape == output_size

    def test_train(self):
        device = torch.device("cpu")
        lapsrn = algorithms.LapSRN(1, 2, device)
        lr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
        hr_images = [np.ones((40, 40), dtype=np.uint8) for _ in range(3)]
        lr_tensors = [transforms.ToTensor()(image) for image in lr_images]
        hr_tensors = [transforms.ToTensor()(image) for image in hr_images]
        loader = data.DataLoader(list(zip(lr_tensors, hr_tensors)))
        lapsrn.train(2, loader, loader, nn.MSELoss(),
                     optim.Adam(lapsrn.get_model().parameters()),
                     stop.NoCondition(), device)

def collate(graphs):
    inputs = [graph[0] for graph in graphs]
    targets = [graph[1] for graph in graphs]
    inputs = dgl.batch(inputs)
    targets = dgl.batch(targets)
    return inputs, targets

class TestSRGCN:
    @pytest.mark.parametrize(["upscale_factor", "input_size", "output_size"],
                             [(2, (20, 20), (40, 40))
                                 , (2, (23, 23), (46, 46))
                                 , (4, (23, 23), (92, 92))
                              ])
    def test_resolve(self, upscale_factor, input_size, output_size):
        device = torch.device("cpu")
        srgcn = algorithms.SRGCN(1, upscale_factor, device)
        image = np.ones(input_size, dtype=np.uint8)
        output = srgcn.resolve(image)
        assert output.shape == output_size

    def test_train(self):
        device = torch.device("cpu")
        srgcn = algorithms.SRGCN(1, 2, device)
        lr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
        hr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
        lr_tensors = [conversion.image_to_graph(image) for image in lr_images]
        hr_tensors = [conversion.image_to_graph(image) for image in hr_images]
        loader = data.DataLoader(list(zip(lr_tensors, hr_tensors)),
                                 collate_fn=collate)
        srgcn.train(2, loader, loader, loss.GraphMSE(),
                    optim.Adam(srgcn.get_model().parameters()),
                    stop.NoCondition(), device)
