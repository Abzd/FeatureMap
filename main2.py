import pickle
import pandas as pd
import numpy as np
import torch
from network import CNN
from index_data import IndexProcessor


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        y = y.long()
        
        X = X.cuda()
        y = y.cuda()

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.cpu().item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.long()
            X = X.cuda()
            y = y.cuda()

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return round(100 * correct, 1)


if __name__ == "__main__":
    ip = IndexProcessor()
    data, y = ip.get_data()

    images = ip.apply_gaf(data)
    print(images.shape)
    print(y.shape)

    train_loader, cv_loader, test_loader = ip.split_data(images, y)
    
    use_gpu = torch.cuda.is_available()
    print('gpu:', use_gpu)
    
    if use_gpu:
        model = CNN(transform='GAF')
        model = model.double().cuda()
        loss_fn = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        max_epochs = 100
        accuracy_list = []
        for t in range(max_epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_loader, model, loss_fn, optimizer)
            acc = test_loop(cv_loader, model, loss_fn)
            accuracy_list.append(acc)
            
            if len(accuracy_list) > 10:
                print(accuracy_list[-10:])
            
            if len(accuracy_list) > 10 and max(accuracy_list[-10:]) == accuracy_list[-10]:
                break

        print('Done!')

        test_loop(test_loader, model, loss_fn)



