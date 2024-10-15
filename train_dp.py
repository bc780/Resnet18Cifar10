import torch
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.distributed as dist
import logging
import torch.nn as nn
from utils.logging import config_logging
from utils.distributed import init_workers

batchsize = 64
epochs= 20
lr = 0.005

def train(device, model, dataLoader, loss_fn,optimizer):
    acc = 0.0
    for image, label in dataLoader:
        pred = model(image)
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        avg_grad(model)
        optimizer.step()
        acc += (pred.argmax(1) == label).type(torch.float).sum().item()
    return 100.0* (acc/len(dataLoader.dataset))

def val(device, model, dataLoader):

    correct = 0

    with torch.no_grad():
        for image, label in dataLoader:
            pred = model(image)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    return 100*(correct/len(dataLoader.dataset))

def avg_grad(model):
    for parameter in model.parameters():
        if type(parameter) is torch.Tensor:
            dist.all_reduce(parameter.grad.data,op=dist.reduce_op.AVG)

# def partition(data, rank, size):
#     data_len = len(data)
#     part_len = data_len/size
#     return data[rank*part_len:(rank+1)*part_len]


if __name__ == "__main__":

    config_logging(verbose=True)
    rank, n_ranks = init_workers("nccl")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    val_data = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())

    gen = torch.Generator().manual_seed(1234)
    train_data_partition = torch.utils.data.random_split(train_data,[1.0/n_ranks]*n_ranks,generator=gen)[rank]
    val_data_partition = torch.utils.data.random_split(val_data,[1.0/n_ranks]*n_ranks,generator=gen)[rank]

    train_dataloader = DataLoader(train_data_partition, batch_size=batchsize, shuffle=True)
    val_dataloader = DataLoader(val_data_partition, batch_size=batchsize, shuffle=True)
    model = models.resnet18()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    for i in range(epochs):
        logging.info("Epoch %i", i+1)
        train_loss = train(device, model, train_dataloader, loss_fn,optimizer)
        logging.info("Training acc: %f", train_loss)
        val_loss = val(device, model, val_dataloader)
        logging.info("Validation acc: %f", val_loss)
