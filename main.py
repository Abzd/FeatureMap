import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from future_data import DataProcessor
from network import CNN
from dataset import ImageData


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        y = y.long()

        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            y = y.long()

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return round(100 * correct, 1)


if __name__ == "__main__":
    train_start = '2012-01-01 01:30:00'
    train_end = '2018-07-01 15:00:00'

    cv_start = '2018-07-02 01:30:00'
    cv_end = '2019-07-01 15:00:00'

    test_start = '2019-07-02 01:30:00'
    test_end = '2021-03-05 15:00:00'

    data_processor = DataProcessor()
    train_loader = data_processor.get_data(train_start, train_end)
    cv_loader = data_processor.get_data(cv_start, cv_end)
    test_loader = data_processor.get_data(test_start, test_end)

    model = CNN(transform='GAF')
    model = model.double()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    max_epochs = 100
    accuracy_list = []
    for t in range(max_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        acc = test_loop(cv_loader, model, loss_fn)
        accuracy_list.append(acc)
        
        if len(accuracy_list) > 8:
            print(accuracy_list[-8:])
        
        if len(accuracy_list) > 8 and max(accuracy_list[-8:]) == accuracy_list[-8]:
            break

    print('Done!')

    test_loop(test_loader, model, loss_fn)

    torch.save(model, 'model.pkl')



