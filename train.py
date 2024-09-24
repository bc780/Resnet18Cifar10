import torch
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import logging
import torch.nn as nn
from utils.logging import config_logging

batchsize = 64
epochs= 20
lr = 0.005

def train(device, model, dataLoader, loss_fn,optimizer):
    acc = 0.0
    for batch, (image, label) in enumerate(dataLoader):
        # image = image.to(device)
        # label = label.to(device)
        pred = model(image)
        loss = loss_fn(pred, label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        acc += (pred.argmax(1) == label).type(torch.float).sum().item()
    return 100.0* (acc/len(dataLoader.dataset))

def val(device, model, dataLoader):

    correct = 0

    with torch.no_grad():
        for image, label in dataLoader:
            # image = image.to(device)
            # label = label.to(device)

            pred = model(image)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    return 100*(correct/len(dataLoader.dataset))


if __name__ == "__main__":

    config_logging(verbose=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
    val_data = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=True)
    model = models.resnet18()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)


    for i in range(epochs):
        logging.info("Epoch %i", i+1)
        train_loss = train(device, model, train_dataloader, loss_fn,optimizer)
        logging.info("Training acc: %f", train_loss)
        val_loss = val(device, model, val_dataloader)
        logging.info("Validation acc: %f", val_loss)
