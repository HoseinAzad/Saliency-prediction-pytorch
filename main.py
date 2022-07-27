import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


import model as model_
from config import *
from utils import *
import dataset as dataset_


def get_model(device):
    model = model_.SalPreNet()
    return model.to(device)


def train(model, dataloader, optimizer, criterion, epoch, device):
    model.train()
    loss_list = []

    for iter, (img, gt_map) in enumerate(dataloader):

        img = img.to(device)
        gt_map = gt_map.to(device)

        output = model(img)
        loss = criterion(output, gt_map)
        loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % round((len(dataloader) / 5)) == 0:
            print(f'[Epoch][Batch] = [{epoch + 1}][{iter}] -> Loss = {np.mean(loss_list):.4f}')

        if iter == len(dataloader)-1:
            # Display raw image, ground truth saliency map, and network output
            display_map((img[0].squeeze(0).data.cpu()).permute(1, 2, 0),
                        output[0].squeeze(0).data.cpu(),
                        gt_map[0].squeeze(0).data.cpu())

    return np.mean(loss_list)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    loss_list = []

    for iterations, (img, gt_map) in enumerate(dataloader):
        img = img.to(device)
        gt_map = gt_map.to(device)

        output = model(img)
        loss = criterion(output, gt_map)
        loss_list.append(loss.item())

    return np.mean(loss_list)


def train_model(data_path, epochs, learning_rate, batch_size, weight_decay, momentum, sc_gamma, sc_step,
                input_width=320, input_height=240,
                output_width=317, output_height=237, split=0.80, title=''):
    # Load dataset
    img_train, img_val, map_train, map_val = load_data(data_path, split)

    print('Number of train samples =', len(img_train))
    print('Number of validation samples =', len(img_val))

    # Instantiate data loaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset_.Dataset(img_train, map_train, input_width, input_height, output_width, output_height),
        batch_size=batch_size, shuffle=False, pin_memory=True)
    val_dataloader = torch.utils.data.DataLoader(
        dataset_.Dataset(img_val, map_val, input_width, input_height, output_width, output_height)
        , batch_size=batch_size, shuffle=False, pin_memory=True)

    print('-' * 40)
    print('Number of train batches =', len(train_dataloader))
    print('Number of validation batches =', len(val_dataloader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('-' * 40)
    print(device, 'is available')

    # Instantiate model
    model = get_model(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sc_step, gamma=sc_gamma)

    best_loss = 0
    loss_list = []
    print('-' * 40, '\nStart Training ....\n')
    for epoch in range(epochs):

        train_loss = train(model, train_dataloader, optimizer, criterion, epoch, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)
        scheduler.step()

        loss_list.append([train_loss, val_loss])

        if val_loss > best_loss:
            torch.save(model, 'best-model.pt')
            best_loss = val_loss

        print(f'\tTrain -> Loss = {train_loss:.4f}')
        print(f'\tValidation -> Loss = {val_loss:.4f}', '\n')

    plot_loss(np.array(loss_list), title)

    best_model = torch.load('best-model.pt')
    return best_model


if __name__ == "__main__":
    model = train_model(data_path = data_path,
                        epochs=epochs,
                        batch_size=bs,
                        learning_rate=lr,
                        weight_decay=wd,
                        momentum=momentum,
                        sc_gamma=sc_g,
                        sc_step=sc_s,
                        split=split)
