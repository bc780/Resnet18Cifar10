import torch
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import logging
import torch.nn as nn

batchsize = 64
epochs=5
lr = 0.001

def train(device, model, dataLoader, loss_fn):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    sum = 0.0
    count = 0.0
    for batch, (image, label) in enumerate(dataLoader):
        image = image.to(device)
        label = label.to(device)

        pred = model(image)
        loss = loss_fn(pred, label)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(image)
            sum += (current/len(dataLoader.dataset))
            count += 1.0
    return sum/count

def val(device, model, dataLoader):
    correct = 0

    with torch.no_grad():
        for image, label in dataLoader:
            image = image.to(device)
            label = label.to(device)

            pred = model(image)
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    return correct/len(dataLoader.dataset)


if __name__ == "__main__":
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

    for i in range(epochs):
        logging.info("Epoch %i", i+1)
        train_loss = train(device, model, train_dataloader, loss_fn)
        logging.info("Training loss: %f", train_loss)
        val_loss = val(device, model, val_dataloader)
        logging.info("Validation loss: %f", val_loss)
