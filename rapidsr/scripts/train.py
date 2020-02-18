import torch
from torch import optim
from torch.utils import data
from rapidsr import algorithms, stop, loss, log
import numpy as np
from torchvision import transforms

timer = log.TimeLogger("", "times.csv")
weights_logger = log.WeightsLogger("", "weights")
loss_logger = log.LossLogger("", "losses.csv")

device = torch.device("cpu")
fsrcnn = algorithms.FSRCNN(1, 2, device)
lr_images = [np.ones((20, 20), dtype=np.uint8) for _ in range(3)]
hr_images = [np.ones((40, 40), dtype=np.uint8) for _ in range(3)]
lr_tensors = [transforms.ToTensor()(image) for image in lr_images]
hr_tensors = [transforms.ToTensor()(image) for image in hr_images]
loader = data.DataLoader(list(zip(lr_tensors, hr_tensors)))
fsrcnn.attach_logger(timer)
fsrcnn.attach_logger(weights_logger)
fsrcnn.attach_logger(loss_logger)
fsrcnn.train(2, loader, loader, loss.MSE(),
             optim.Adam(fsrcnn.get_model().parameters()),
             stop.NoCondition(), device)
