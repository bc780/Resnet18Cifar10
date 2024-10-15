import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import logging
import torch.nn as nn
from utils.logging import config_logging
from utils.distributed import init_workers

batchsize = 40
epochs= 20
lr = 0.01

def train(device, model, dataLoader, loss_fn,optimizer):
    model.train()
    acc = 0.0
    for image, label in dataLoader:
        image = image.to(device)
        label = label.to(device)
        if rank == 0:
            temp = model(image,rank)
            temp.retain_grad()
            # logging.info("SHAPE: %i %i %i %i", temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3])
            dist.send(temp,dst=1)
        if rank == 1:
            temp = torch.zeros([40,128,28,28],requires_grad=True).to(device)
            dist.recv(temp, src=0)
            temp.retain_grad()
            # logging.info("Sum %f", temp.sum())
            pred = model(temp,rank)      
            
            loss = loss_fn(pred, label)
            optimizer.zero_grad()
            loss.backward()
            dist.send(temp.grad, dst=0)
            optimizer.step()
            acc += (pred.argmax(1) == label).type(torch.float).sum().item()
            # logging.info("Pred: %i", len(pred.argmax(1)))
            # logging.info("Label: %i", len(label))
            # logging.info("Acc: %f", acc)
        if rank == 0:
            optimizer.zero_grad()
            temp_grad = torch.zeros_like(temp).to(device)
            dist.recv(temp_grad, src=1)
            temp.backward(gradient=temp_grad)
            optimizer.step()

        dist.barrier()
    return 100.0* (acc/len(dataLoader.dataset))

def val(device, model, dataLoader):

    correct = 0

    with torch.no_grad():
        for image, label in dataLoader:
            image = image.to(device)
            label = label.to(device)
            if rank == 0:
                temp = model(image,rank)
                dist.send(temp,dst=1)
            if rank == 1:
                temp = torch.zeros([40,128,28,28]).to(device)
                dist.recv(temp, src=0)
                pred = model(temp,rank)            
                correct += (pred.argmax(1) == label).type(torch.float).sum().item()
            dist.barrier()
    return 100*(correct/len(dataLoader.dataset))


class ModelParallel(nn.Module):
    def __init__(self, rank):
        super(ModelParallel, self).__init__()
        
        model = models.resnet18(weights=None)
        
        if rank == 0:
            self.seq1 = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2
            ).to('cuda')
        if rank == 1:
            self.seq2 = nn.Sequential(
                model.layer3,
                model.layer4,
                model.avgpool
            ).to('cuda')

            self.fc = model.fc.to('cuda')

    def forward(self, x, rank):
        if rank == 0:
            x = self.seq1(x)
            return x
        if rank == 1:
            x = self.seq2(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
def run(rank, size):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])    
    train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    val_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batchsize, shuffle=True)
    model = ModelParallel(rank)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(epochs):
        logging.info("Epoch %i", i+1)
        train_loss = train(device, model, train_dataloader, loss_fn,optimizer)
        logging.info("Training acc: %f", train_loss)
        val_loss = val(device, model, val_dataloader)
        logging.info("Validation acc: %f", val_loss)

if __name__ == "__main__":

    config_logging(verbose=True)
    rank, n_ranks = init_workers("nccl")
    run(rank, n_ranks)
    
