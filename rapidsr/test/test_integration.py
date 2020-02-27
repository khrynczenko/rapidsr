import torch
from torch import optim
from torch.utils import data
from rapidsr import algorithms, stop, loss, stop, conversion
import dgl

import numpy as np
from torchvision import transforms


def test_training_cnn():
    device = torch.device("cpu")
    fsrcnn = algorithms.FSRCNN(1, 2, device)
    lr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
    hr_images = [np.ones((40, 40), dtype=np.uint8) for _ in range(3)]
    lr_tensors = [transforms.ToTensor()(image) for image in lr_images]
    hr_tensors = [transforms.ToTensor()(image) for image in hr_images]
    loader = data.DataLoader(list(zip(lr_tensors, hr_tensors)))
    fsrcnn.train(2, loader, loader, loss.MSE(),
                 optim.Adam(fsrcnn.get_model().parameters()),
                 stop.NoCondition(), device)


def collate_graphs(batch):
    input_graphs = [entry[0] for entry in batch]
    input_graphs_size = [g.size for g in input_graphs]
    output_graphs = [entry[1] for entry in batch]
    output_graphs_size = [g.size for g in output_graphs]
    batched_input_graphs = dgl.batch(input_graphs)
    batched_input_graphs.size = lambda: input_graphs_size
    batched_output_graphs = dgl.batch(output_graphs)
    batched_output_graphs.size = lambda: output_graphs_size
    return batched_input_graphs, batched_output_graphs


def test_training_graph():
    device = torch.device("cpu")
    fsrcnn = algorithms.SRGCN(1, 2, device)
    lr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
    hr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
    lr_graphs = [conversion.image_to_graph(image) for image in lr_images]
    hr_graphs = [conversion.image_to_graph(image) for image in hr_images]

    loader = data.DataLoader(list(zip(lr_graphs, hr_graphs)),
                             collate_fn=collate_graphs)
    fsrcnn.train(2, loader, loader, loss.GraphMSE(),
                 optim.Adam(fsrcnn.get_model().parameters()),
                 stop.NoCondition(), device)
