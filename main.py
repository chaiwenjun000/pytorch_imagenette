import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models
from PIL import Image

class ImagenetteDataset(Dataset):
    def __init__(self, X, y, transforms):
        self.X = X
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        img = self.transforms(img)

        return img, label


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[y]


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    batch_size = 64
    test_size = 0.05
    image_size = 160
    out_features = 10
    lr = 0.001
    epochs = 100
    pretrained = False

    X = np.load('X_train.npy')
    y = np.load('y_train.npy')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=0)

    X_train = X_train.astype('uint8')
    X_test = X_test.astype('uint8')
    y_train = y_train.astype('int64')
    y_test = y_test.astype('int64')

    transforms_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size=image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((110/255,117/255,118/255),(76/255,71/255,72/255))
    ])

    transforms_test = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(), transforms.Normalize((110/255,117/255,118/255),(76/255,71/255,72/255))])

    train_dataset = ImagenetteDataset(X_train, y_train, transforms_train)
    test_dataset = ImagenetteDataset(X_test, y_test, transforms_test)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(in_features=2048, out_features=out_features)  # change for imagenette

    if use_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        global_step = 0
        current_loss = 0
        correct_train_preds = 0.0
        total_train_preds = 0.0
        for idx, batch_data in enumerate(train_dataloader):
            batch_x, batch_y = batch_data

            if use_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            model.zero_grad()
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()

            total_train_preds += batch_y.shape[0]
            correct_train_preds += torch.sum(torch.argmax(predictions, dim=1) == batch_y).item()

            global_step += 1
            if global_step % 80 == 0:
                print('Epoch: {}, global_step: {}, current_loss: {}, accuracy: {}'.format(epoch, global_step, current_loss,
                                                                                          correct_train_preds / total_train_preds))
                current_loss = 0
                correct_train_preds = 0.0
                total_train_preds = 0.0

        model.eval()

        testing_loss = 0
        correct_test_preds = 0.0
        total_test_preds = 0.0
        for batch_data in test_dataloader:
            batch_x, batch_y = batch_data

            if use_gpu:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)

            total_test_preds += batch_y.shape[0]
            correct_test_preds += torch.sum(torch.argmax(predictions, dim=1) == batch_y).item()

            testing_loss += loss.item()

        print('Testing Set - Epoch: {}, loss: {}, accuracy: {}'.format(epoch, testing_loss, correct_test_preds / total_test_preds))

        model.train()
