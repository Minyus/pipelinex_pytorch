import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.datasets import MNIST


def generate_datasets(parameters={}):
    data_transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_dataset = MNIST(download=True, root=".", transform=data_transform, train=True)
    val_dataset = MNIST(download=False, root=".", transform=data_transform, train=False)
    return train_dataset, val_dataset


def _final_transform(tensor):
    batch_pred_arr = torch.argmax(tensor, dim=1).numpy()
    return batch_pred_arr


def infer(model, test_dataset, parameters={}):
    test_batch_size = parameters.get("test_batch_size", 1)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    pred_list = []
    with torch.no_grad():
        for images in test_loader:
            if isinstance(images, (tuple, list)):
                images = images[0]
            outputs = model(images)
            batch_pred_arr = _final_transform(outputs)
            pred_list.append(batch_pred_arr)
    pred_arr = np.concatenate(pred_list, axis=None)
    return pd.DataFrame(dict(pred=pred_arr))
