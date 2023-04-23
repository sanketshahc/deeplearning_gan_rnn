import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchvision
import asgn_1 as SANKETNET
import torchvision.transforms.transforms as transforms
import pickle
import torchvision.models as models
import torchvision.datasets as datasets
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import os
import PIL.Image as pil
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Action
import re
import time
from hypes_gan import *
Lambda = transforms.Lambda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#########################################

# Params
paths = {
    1: "text/all_merged.txt",
    2: 'text/test_neg_merged.txt',
    3: 'text/test_pos_merged.txt',
    4: 'text/train_neg_merged.txt',
    5: "text/train_pos_merged.txt"
}


### Helpers

def save_bin(name, f_object):
    stamp = int(time.time())
    if not os.path.exists('pickled_binaries/'):
        os.makedirs('pickled_binaries/')
    try:
        f_object = f_object.to('cpu')
        name = f'{name}_{stamp}.pt'
        torch.save(f_object, f"pickled_binaries/{name}")
    except:
        name = f'{name}_{stamp}.bin'
        file = open(f"pickled_binaries/{name}", "wb")
        pickle.dump(f_object, file)
        file.close()

def strip_split(path):
    embed_words = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.lower()
            line = re.sub("\W", " ", line)
            line = re.sub("(\s.{2}\s)|(\s.{1}\s.{1}\s)|(\s.{1}\s)", " ", line)
            line = re.sub("\s+", ",", line)
            line = re.sub(",$", "", line)
            arr_line = line.split(',')
            embed_words.append(arr_line)
    return embed_words


def vocabulary_builder(embed_words):
    hist = {}
    ind = 0
    for line in embed_words:
        for word in line:
            if word not in hist.keys():
                hist[word] = ind
                ind += 1
    return hist


def index_cut_pad(embed_words):
    embeddings = []
    for arr_line in embed_words:
        arr_line = arr_line[:400]
        x = len(arr_line)
        for i, j in enumerate(arr_line):
            # print(i,j)
            arr_line[i] = vocabulary[j]
        assert (type(arr_line[0] == int)), arr_line
        if x < 400:
            arr_line = [0 for i in range(400 - x)] + arr_line
        assert (len(arr_line) == 400), len(arr_line)
        embeddings.append(arr_line)

    return np.array(embeddings)


def transform_fmnist(img):
    img = transforms.ToTensor()(img)  # 0 to 1 tensor
    img = transforms.Normalize(mean=0.1307, std=0.3081)(img)
    return img

## FROM GAN CLASSES
def peak(self, z, name='x'):
    inv_normalize = transforms.Normalize(
        mean=[-0.1307 / 0.3081],
        std=[1 / 0.3081]
    )
    g = self.generator(z)
    g = g.reshape(batch_size, 1, 28, 28)
    g = inv_normalize(g)
    tiles = self.tile_and_print(g, 8, 8)
    tiles = tiles.permute(1, 2, 0)
    tiles = tiles.cpu().detach().numpy()
    tiles = tiles.squeeze()
    plt.figure(figsize=(80, 40))
    plt.imshow(tiles, interpolation='bilinear', cmap='gray')
    plt.savefig(f'./plots/peak_{name}_{int(time.time())}.png')
    plt.close()

## FROM GAN CLASSES
def tile_and_print(self, input, tiles_height, tiles_width, padding=1):  # taken from asgn2
    """
    expecting a 4d weight tensor. (chan_out, chan_in, h, w). permute for matplot plot.
    This function uses permute to compose the filter map....
    """
    device = input.device
    p = padding
    w = input
    assert len(w.shape) == 4, w.shape
    co, ci, he, wi = w.shape
    assert he * wi == xout_size
    if padding:
        w = torch.cat((w, torch.ones((co, ci, p, wi), device=device)), dim=-2)
        w = torch.cat((w, torch.ones((co, ci, he + p, p), device=device)), dim=-1)
        co, ci, he, wi = w.shape
    w = w.permute(1, 2, 3, 0)
    w = w.reshape(ci, he, wi, tiles_height, tiles_width)
    w = w.permute(0, 3, 1, 4, 2)
    w = w.reshape(ci, he * tiles_height, tiles_width * wi)
    return w

### Problem 1

class GRURnnet(nn.Module):
    def __init__(self, voc_size, embed_size, seq_len, hidden_size):  # input size = seq size
        super(GRURnnet, self).__init__()
        self.voc_size = voc_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(self.voc_size, self.embed_size)
        self.gru = nn.GRU(
            input_size=self.embed_size, hidden_size=self.hidden_size,batch_first=True
        )
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = x.squeeze()
        x = self.embed(x)
        x = self.gru(x)
        x = x[0][:, -1, :]
        x = self.linear(x)
        x = nn.Sigmoid()(x)
        return x.float()

    def init_hidden(self):
        self.h0 = torch.zeros(batch_size, self.hidden_size)


class MLP_net(nn.Module):
    def __init__(self, seq_len, voc_size):
        super(MLP_net, self).__init__()
        self.embed_size = 16
        self.seq_len = seq_len
        self.hidden_size = 64
        self.embed = nn.Embedding(voc_size, self.embed_size)
        self.linear1 = nn.Linear(self.embed_size * self.seq_len, 1)

    def forward(self, x):
        x = self.embed(x)
        x = x.reshape(batch_size, self.embed_size * self.seq_len)
        x = self.linear1(x)
        x = nn.Sigmoid()(x)
        return x.float()


def batches_loop(loader, model, criterion, optimizer, is_val=False):
    if type(model) == GRURnnet:
        model.init_hidden()
    batch_count = 0
    loss_total = 0
    count_correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.float()
        y = y.to(device)
        batch_count += 1
        if is_val:
            with torch.no_grad():
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss_total += loss.item()
        else:
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss_total += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        if batch_count % 50 == 0:
            print(batch_count, ' batches complete')
        # Accuracy Comp
        count_correct += (y == torch.round(y_hat)).sum()
    accuracy = (count_correct / (batch_count * batch_size))
    return loss_total, accuracy


def problem_1(network, train_loader, test_loader):
    network.to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=w)  # is
    criterion = nn.BCELoss()
    network.train()
    count_epoch = 0
    accuracies = []
    accuracies_test = []
    loss_testing = []
    loss_training = []
    for epoch in range(epochs):
        count_epoch += 1
        print('EPOCH:', count_epoch)
        network.train()
        training_batches = batches_loop(train_loader, network, criterion, optimizer)
        loss_training.append(training_batches[0])
        network.eval()
        testing_batches = batches_loop(test_loader, network, criterion, optimizer, True)
        loss_testing.append(testing_batches[0])
        accuracy = training_batches[1]
        accuracies.append(accuracy)
        accuracy_test = testing_batches[1]
        accuracies_test.append(accuracy_test)
        print('training_loss', training_batches[0], 'accuracy', accuracy)
        print('testing_loss', testing_batches[0], 'accuracy_test', accuracy_test)
        metrics = (loss_training, loss_testing, accuracies, accuracies_test)
    save_bin(f'rnn', network)
    save_bin(f'rnn_metrics', metrics)
    return network, metrics


def data_prep():
    testing_neg = index_cut_pad(strip_split(paths[2]))
    testing_pos = index_cut_pad(strip_split(paths[3]))
    testing_combined = np.vstack((testing_neg, testing_pos))
    testing_target = np.vstack((
        np.zeros((testing_neg.shape[0], 1), ),
        np.ones((testing_pos.shape[0], 1), )
    ))
    testing_combined = testing_combined.reshape(
        testing_combined.shape[0], testing_combined.shape[1]
    )
    training_neg = index_cut_pad(strip_split(paths[4]))
    training_pos = index_cut_pad(strip_split(paths[5]))
    training_combined = np.vstack((training_neg, training_pos))
    training_target = np.vstack((
        np.zeros((training_neg.shape[0], 1), ),
        np.ones((training_pos.shape[0], 1), )
    ))
    training_combined = training_combined.reshape(
        training_combined.shape[0], training_combined.shape[1]
    )
    train_data = TensorDataset(torch.from_numpy(training_combined), torch.from_numpy(
        training_target))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_data = TensorDataset(torch.from_numpy(testing_combined), torch.from_numpy(
        testing_target))
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
    voc_size = len(vocabulary)
    input_size = training_combined.shape[-1]
    return train_loader, test_loader, voc_size, input_size


### Problem 2

fmnist = torchvision.datasets.FashionMNIST(
    root="./", train=True,
    transform=transform_fmnist, download=True)

fmnist_loader = torch.utils.data.DataLoader(
    dataset=fmnist,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)

fmnist_test = torchvision.datasets.FashionMNIST(
    root="./",
    transform=transform_fmnist, download=True)

fmnist_test_loader = torch.utils.data.DataLoader(
    dataset=fmnist_test,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)


class Adversary(nn.Module):
    def __init__(self, z_size, hs1, hs2, hs3, xout_size):
        super(Adversary, self).__init__()
        self.lin1 = nn.Linear(z_size, hs1)
        self.lin2 = nn.Linear(hs1, hs2)
        self.lin3 = nn.Linear(hs2, hs3)
        self.output = nn.Linear(hs3, xout_size)
        self.a1 = nn.LeakyReLU(.2)  # CONSIDER LEAKY RELU
        self.a2 = nn.Tanh()
        pass

    def forward(self, z):
        z = self.lin1(z)
        z = self.a1(z)
        z = self.lin2(z)
        z = self.a1(z)
        z = self.lin3(z)
        z = self.a1(z)
        z = self.output(z)
        z = self.a2(z)
        return z  # generated output


class Discriminator(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(input_size, hs1)
        self.lin2 = nn.Linear(hs1, hs2)
        self.lin3 = nn.Linear(hs2, hs3)
        self.output = nn.Linear(hs3, 1)
        self.a1 = nn.LeakyReLU(.2)  # CONSIDER LEAKY RELU
        self.a2 = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = self.a1(x)
        x = self.lin2(x)
        x = self.a1(x)
        x = self.lin3(x)
        x = self.a1(x)
        x = self.output(x)
        x = self.a2(x)
        return x  # real/fake score

    def peak_weights(self):
        for each in self.parameters():
            print()


class CriticWasserstein(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3):
        super(CriticWasserstein, self).__init__()
        self.lin1 = nn.Linear(input_size, hs1)
        self.lin2 = nn.Linear(hs1, hs2)
        self.lin3 = nn.Linear(hs2, hs3)
        self.output = nn.Linear(hs3, 1)
        self.a1 = nn.LeakyReLU(.2)  # CONSIDER LEAKY RELU
        self.a2 = nn.Sigmoid()

    def forward(self, x):
        x = self.lin1(x)
        x = self.a1(x)
        x = self.lin2(x)
        x = self.a1(x)
        x = self.lin3(x)
        x = self.a1(x)
        x = self.output(x)
        # x = self.a2(x)
        return x  # real/fake score

    def peak_weights(self):
        for each in self.parameters():
            print()


class DiscriminatorClassifier(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3):
        super(DiscriminatorClassifier, self).__init__()
        self.lin1 = nn.Linear(input_size, hs1)
        self.lin2 = nn.Linear(hs1, hs2)
        self.lin3 = nn.Linear(hs2, hs3)
        self.output = nn.Linear(hs3, 10)
        self.a1 = nn.LeakyReLU(.2)  # CONSIDER LEAKY RELU

    def forward(self, x):
        x = self.lin1(x)
        x = self.a1(x)
        x = self.lin2(x)
        x = self.a1(x)
        x = self.lin3(x)
        x = self.a1(x)
        x = self.output(x)
        return x  # real/fake score


class GAN(nn.Module):
    def __init__(self, criterion):
        super(GAN, self).__init__()
        self.generator = Adversary(z_size, hs_g1, hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator(xout_size, hs_d1, hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train()
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            weight_decay=w_g,
            betas=(beta1_g, beta2_g)
        )
        self.optim_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            weight_decay=w_d,
            betas=(beta1_d, beta2_d)
        )

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth):
        score = score.to(device)
        truth = truth.to(device)
        return self.criterion(score, truth)  # takes mean reduction

    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        for x, _ in fmnist_loader:
            batch_count += 1
            x = x.squeeze()
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            z = torch.randn(batch_size, z_size)
            z = z.to(device)

            d_g = self.discriminator(
                self.generator(z).detach()
            )
            y_dg = torch.zeros(batch_size, 1)
            y_dg = y_dg.to(device)
            loss_dg = self.loss(d_g, y_dg)
            d_x = self.discriminator(x)
            y_dx = torch.ones(batch_size, 1)
            y_dx = y_dx.to(device)
            loss_dx = self.loss(d_x, y_dx)
            loss_dt = loss_dx + loss_dg
            loss_total_d += loss_dt.item()
            optim_d.zero_grad()
            loss_dt.backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), c_d
            )
            optim_d.step()

            if batch_count % rd == 0:
                for i in range(rg):
                    z = torch.randn(batch_size, z_size)
                    z = z.to(device)
                    g = self.generator(z)
                    y_g = self.discriminator(g)
                    y_gt = torch.ones(batch_size, 1)
                    y_gt = y_gt.to(device)
                    y_g = y_g.to(device)
                    loss_g = self.loss(y_g, y_gt)
                    optim_g.zero_grad()
                    loss_total_g += loss_g.item()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), c_g
                    )
                    optim_g.step()

            if batch_count % 200 == 0:
                print(batch_count,
                      f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

        for i, each in enumerate(self.generator.parameters()):
            print('generator weight norms', torch.norm(each)) if i % 2 == 0 else None
        for i, each in enumerate(self.discriminator.parameters()):
            print('discriminator weight norms', torch.norm(each)) if i % 2 == 0 else None
        self.loss_totals_g.append(loss_total_g)
        self.loss_totals_d.append(loss_total_d)
        self.score_g.append(d_g.detach().mean().item())
        self.score_d.append(d_x.detach().mean().item())
        return

    def train_gan(self):
        count_epoch = 0
        for epoch in range(epochs):
            count_epoch += 1
            print('EPOCH:', count_epoch)
            self.batches_loop()
            self.peak(self.seed, name='ss')

        save_bin(f'gan', self)

    def peak(self, z, name='x'):
        ## SEE HELPERS
        pass

    def tile_and_print(self, input, tiles_height, tiles_width, padding=1):  # taken from asgn2
        ## SEE HELPERS
        pass


class GAN_MSE(nn.Module):
    def __init__(self, criterion):
        super(GAN_MSE, self).__init__()
        self.generator = Adversary(z_size, hs_g1, hs_g2, hs_g3, xout_size)
        self.criterion = criterion
        self.to(device)
        self.train()
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_d = []
        self.score_g = []
        self.optim_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            weight_decay=w_g,
            betas=(beta1_g, beta2_g)
        )

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth):
        score = score.to(device)
        truth = truth.to(device)
        return self.criterion(score, truth)

    def batches_loop(self):
        optim_g = self.optim_g
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        for x, _ in fmnist_loader:
            batch_count += 1

            # data_rinse
            x = x.squeeze()
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            z = torch.randn(batch_size, z_size)
            z = z.to(device)

            y_g = self.generator(z)
            y_g = y_g.to(device)

            loss_g = self.loss(y_g, x)
            optim_g.zero_grad()
            loss_total_g += loss_g.item()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), c_g)
            optim_g.step()

            if batch_count % 200 == 0:
                print(batch_count,
                      f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

        for i, each in enumerate(self.generator.parameters()):
            print('generator weight norms', torch.norm(each)) if i % 2 == 0 else None
        self.loss_totals_g.append(loss_total_g)

        return

    def train_gan(self):
        count_epoch = 0
        for epoch in range(epochs):
            count_epoch += 1
            print('EPOCH:', count_epoch)
            self.batches_loop()
            self.peak(self.seed, name='ss')

        save_bin(f'gan', self)

    def peak(self, z, name='x'):
        ## SEE HELPERS
        pass

    def tile_and_print(self, input, tiles_height, tiles_width, padding=1):  # taken from asgn2
        ## see helpers
        pass


# overall network system module....will have training funcitons embedded, like sanketnet.
class GAN_Wass(nn.Module):
    def __init__(self, criterion):
        super(GAN_Wass, self).__init__()
        self.generator = Adversary(z_size, hs_g1, hs_g2, hs_g3, xout_size)
        self.discriminator = CriticWasserstein(xout_size, hs_d1, hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train()  # NEcessary? maybe not
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=learning_rate_g,
                                        weight_decay=w_g, betas=(beta1_g, beta2_g))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_d,
                                        weight_decay=w_d, betas=(beta1_d, beta2_d))

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth, d):
        if isinstance(score, int) or isinstance(score, float):
            score = torch.tensor(score)
        if isinstance(truth, int) or isinstance(truth, float):
            truth = torch.tensor(truth)
        score = score.to(device)
        truth = truth.to(device)
        return self.criterion(score, truth, d)

    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        for x, _ in fmnist_loader:
            batch_count += 1
            # data_rinse
            x = x.squeeze()
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)
            d_g = self.discriminator(self.generator(z).detach())
            d_x = self.discriminator(x)
            loss_dx = self.loss(d_x, d_g, d=True)
            loss_total_d += loss_dx.item()
            optim_d.zero_grad()
            loss_dx.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), c_d)
            optim_d.step()
            for p, each in enumerate(self.discriminator.parameters()):
                if p % 2 == 0:
                    each.data.clamp_(-c_w, c_w)

            if batch_count % rd == 0:
                for i in range(rg):
                    z = torch.randn(batch_size, z_size)  # rand latent
                    z = z.to(device)

                    g = self.generator(z)
                    y_g = self.discriminator(g)
                    y_g = y_g.to(device)

                    # generator loss, backward, step
                    loss_g = self.loss(0, y_g, d=False)
                    optim_g.zero_grad()
                    loss_total_g += loss_g.item()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), c_g)
                    optim_g.step()

            if batch_count % 200 == 0:
                print(batch_count,
                      f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

        for i, each in enumerate(self.generator.parameters()):
            print('generator weight norms', torch.norm(each)) if i % 2 == 0 else None
        for i, each in enumerate(self.discriminator.parameters()):
            print('discriminator weight norms', torch.norm(each)) if i % 2 == 0 else None
        self.loss_totals_g.append(loss_total_g)
        self.loss_totals_d.append(loss_total_d)
        self.score_g.append(d_g.detach().mean().item())
        self.score_d.append(d_x.detach().mean().item())
        return

    def train_gan(self):
        count_epoch = 0
        for epoch in range(epochs):
            count_epoch += 1
            print('EPOCH:', count_epoch)
            self.batches_loop()
            self.peak(self.seed, name='ss')

        save_bin(f'gan', self)

    def peak(self, z, name='x'):
        inv_normalize = transforms.Normalize(
            mean=[-0.1307 / 0.3081],
            std=[1 / 0.3081]
        )
        g = self.generator(z)
        g = g.reshape(batch_size, 1, 28, 28)
        g = inv_normalize(g)
        tiles = self.tile_and_print(g, 8, 8)
        tiles = tiles.permute(1, 2, 0)
        tiles = tiles.cpu().detach().numpy()
        tiles = tiles.squeeze()
        plt.figure(figsize=(80, 40))
        plt.imshow(tiles, interpolation='bilinear', cmap='gray')
        plt.savefig(f'./plots/peak_{name}_{int(time.time())}.png')
        plt.close()  # shows up as green?

    def tile_and_print(self, input, tiles_height, tiles_width, padding=1):  # taken from asgn2
        """
        expecting a 4d weight tensor. (chan_out, chan_in, h, w). permute for matplot plot.
        This function uses permute to compose the filter map....
        """
        device = input.device
        p = padding
        w = input
        assert len(w.shape) == 4, w.shape
        co, ci, he, wi = w.shape
        assert he * wi == xout_size
        if padding:
            w = torch.cat((w, torch.ones((co, ci, p, wi), device=device)), dim=-2)
            w = torch.cat((w, torch.ones((co, ci, he + p, p), device=device)), dim=-1)
            co, ci, he, wi = w.shape
        w = w.permute(1, 2, 3, 0)
        w = w.reshape(ci, he, wi, tiles_height, tiles_width)
        w = w.permute(0, 3, 1, 4, 2)
        w = w.reshape(ci, he * tiles_height, tiles_width * wi)
        return w


class GAN_lsq(nn.Module):
    def __init__(self, criterion):
        super(GAN_lsq, self).__init__()
        self.generator = Adversary(z_size, hs_g1, hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator(xout_size, hs_d1, hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train()
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            weight_decay=w_g,
            betas=(beta1_g, beta2_g)
        )
        self.optim_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            weight_decay=w_d,
            betas=(beta1_d, beta2_d)
        )

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth, d):
        if isinstance(score, int) or isinstance(score, float):
            score = torch.tensor(score)
        if isinstance(truth, int) or isinstance(truth, float):
            truth = torch.tensor(truth)
        score = score.to(device)
        truth = truth.to(device)
        return self.criterion(score, truth, d)  # takes mean reduction

    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        for x, _ in fmnist_loader:
            batch_count += 1
            x = x.squeeze()
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            z = torch.randn(batch_size, z_size)
            z = z.to(device)

            d_g = self.discriminator(
                self.generator(z).detach()
            )
            d_x = self.discriminator(x)
            loss_dx = self.loss(d_x, d_g, d=True)
            loss_dt = loss_dx  # +loss_dg
            loss_total_d += loss_dt.item()
            optim_d.zero_grad()
            loss_dt.backward()
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), c_d
            )
            optim_d.step()

            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)
            g = self.generator(z)
            y_g = self.discriminator(g)  # fake score
            y_g = y_g.to(device)
            loss_g = self.loss(0, y_g, d=False)
            optim_g.zero_grad()
            loss_total_g += loss_g.item()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), c_g)
            optim_g.step()
            if batch_count % 200 == 0:
                print(batch_count,
                      f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

        for i, each in enumerate(self.generator.parameters()):
            print('generator weight norms', torch.norm(each)) if i % 2 == 0 else None
        for i, each in enumerate(self.discriminator.parameters()):
            print('discriminator weight norms', torch.norm(each)) if i % 2 == 0 else None
        self.loss_totals_g.append(loss_total_g)
        self.loss_totals_d.append(loss_total_d)
        self.score_g.append(d_g.detach().mean().item())
        self.score_d.append(d_x.detach().mean().item())
        return

    def train_gan(self):
        count_epoch = 0
        for epoch in range(epochs):
            count_epoch += 1
            print('EPOCH:', count_epoch)
            self.batches_loop()
            self.peak(self.seed, name='ss')

        save_bin(f'gan', self)

    def peak(self, z, name='x'):
        ## SEE HELPER FNS
        pass

    def tile_and_print(self, input, tiles_height, tiles_width, padding=1):  # taken from asgn2
        # SEE HELPERS
        pass


class FMnist_classifier(nn.Module):
    def __init__(self):
        super(FMnist_classifier, self).__init__()
        self.discriminator = DiscriminatorClassifier(xout_size, hs_d1, hs_d2, hs_d3)
        self.criterion = nn.CrossEntropyLoss()
        self.to(device)
        self.loss_totals = []
        self.loss_totals_test = []
        self.accuracy = []
        self.accuracy_test = []
        self.histogram = []
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_d)

    def forward(self, input):
        return self.discriminator(input)

    def loss(self, y_hat, y, d):
        y_hat = y_hat.to(device)
        y = y.to(device)
        return self.criterion(y_hat, y, d)

    def batches_loop(self, loader, is_val=False):
        model = self.forward
        optimizer = self.optimizer
        criterion = self.criterion
        batch_count = 0
        loss_total = 0
        count_correct = 0
        for x, y in loader:
            batch_count += 1
            x = x.squeeze()
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            y = y.to(device)
            if is_val:
                with torch.no_grad():
                    y_hat = model(x)
                    loss = criterion(y_hat, y)
                    loss_total += loss.item()
            else:
                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss_total += loss.item()
                loss.backward()
                optimizer.step()
            if batch_count % 200 == 0:
                print(batch_count, f'batches complete')

            y_hat_arg = y_hat.argmax(dim=-1)
            self.histogram += y_hat_arg.tolist()
            count_correct += (y == y_hat_arg).sum()

        accuracy = (count_correct / (batch_count * batch_size))
        if is_val:
            self.loss_totals_test.append(loss_total)
            self.accuracy_test.append(accuracy)
        else:
            self.loss_totals.append(loss_total)
            self.accuracy.append(accuracy)
        print(f'EPOCH complete, loss: {loss_total}, accuracy: {accuracy}')
        return

    def train_classifier(self):
        network = self.discriminator
        count_epoch = 0
        for epoch in range(epochs):
            count_epoch += 1
            print('EPOCH:', count_epoch)
            network.train()
            print('Training...')
            self.batches_loop(fmnist_loader)
            network.eval()
            print('Validating...')
            self.batches_loop(fmnist_test_loader, is_val=True)
        metrics = (self.loss_totals, self.loss_totals_test, self.accuracy, self.accuracy_test)
        save_bin(f'class', network)
        save_bin(f'class_metrics', metrics)


class GAN_unrolled(nn.Module):
    def __init__(self, criterion):
        super(GAN_unrolled, self).__init__()
        self.generator = Adversary(z_size, hs_g1, hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator(xout_size, hs_d1, hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train()
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=learning_rate_g,
                                        weight_decay=w_g, betas=(beta1_g, beta2_g))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_d,
                                        weight_decay=w_d, betas=(beta1_d, beta2_d))

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth):
        score = score.to(device)
        truth = truth.to(device)
        return self.criterion(score, truth)

    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        fmnist_iter = iter(fmnist_loader)
        for x, _ in fmnist_iter:
            batch_count += 1
            x = x.squeeze()
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)

            d_g = self.discriminator(self.generator(z).detach())
            y_dg = torch.zeros(batch_size, 1)
            y_dg = y_dg.to(device)
            loss_dg = self.loss(d_g, y_dg)
            d_x = self.discriminator(x)
            y_dx = torch.ones(batch_size, 1)
            y_dx = y_dx.to(device)

            # discriminator loss, backward, step
            loss_dx = self.loss(d_x, y_dx)
            loss_dt = loss_dx + loss_dg
            loss_total_d += loss_dt.item()
            optim_d.zero_grad()
            loss_dt.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), c_d)
            optim_d.step()

            if batch_count % rd == 0:
                assert rolls > 0, 'unrolled gan needs rolls'
                for k in range(rolls):
                    batch_count += 1
                    try:
                        _x, _ = next(fmnist_iter)
                    except StopIteration:
                        "Iteration Finished, breaking"
                        break
                    _x = _x.squeeze()  #
                    _x = _x.reshape(batch_size, xout_size)
                    _x = _x.to(device)
                    unrolled_discriminator = type(self.discriminator)(
                        xout_size, hs_d1, hs_d2,hs_d3)  # get a new instance
                    unrolled_discriminator = unrolled_discriminator.to(device)
                    unrolled_discriminator.load_state_dict(
                        self.discriminator.state_dict())
                    z = torch.randn(batch_size, z_size)  # rand latent
                    z = z.to(device)
                    u_g = unrolled_discriminator(
                        self.generator(z).detach()
                    )
                    y_ug = torch.zeros(batch_size, 1)
                    y_ug = y_ug.to(device)
                    loss_ug = self.loss(u_g, y_ug)
                    u_x = unrolled_discriminator(_x)
                    y_ux = torch.ones(batch_size, 1)
                    y_ux = y_ux.to(device)
                    # d = d_g + d_x
                    # discriminator loss, backward, step
                    loss_ux = self.loss(u_x, y_ux)
                    loss_ut = loss_ux + loss_ug
                    optim_d.zero_grad()
                    loss_ut.backward()
                    torch.nn.utils.clip_grad_norm_(
                        unrolled_discriminator.parameters(), c_d
                    )
                    optim_d.step()

                for i in range(rg):
                    # generator forward
                    z = torch.randn(batch_size, z_size)  # rand latent
                    z = z.to(device)
                    g = self.generator(z)
                    y_g = unrolled_discriminator(g)  # fake score
                    y_gt = torch.ones(batch_size, 1)  # target
                    y_gt = y_gt.to(device)
                    y_g = y_g.to(device)

                    # generator loss, backward, step
                    loss_g = self.loss(y_g, y_gt)
                    optim_g.zero_grad()
                    loss_total_g += loss_g.item()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.generator.parameters(), c_g
                    )
                    optim_g.step()

            if batch_count % 200 == 0:
                print(batch_count,
                      f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

        for i, each in enumerate(self.generator.parameters()):
            print('generator weight norms', torch.norm(each)) if i % 2 == 0 else None
        for i, each in enumerate(self.discriminator.parameters()):
            print('discriminator weight norms', torch.norm(each)) if i % 2 == 0 else None
        # self.peak(z, name='train')
        self.loss_totals_g.append(loss_total_g)
        self.loss_totals_d.append(loss_total_d)
        self.score_g.append(d_g.detach().mean().item())
        self.score_d.append(d_x.detach().mean().item())
        return

    def train_gan(self):
        count_epoch = 0
        for epoch in range(epochs):
            count_epoch += 1
            print('EPOCH:', count_epoch)
            self.batches_loop()
            self.peak(self.seed, name='ss')

        save_bin(f'gan', self)

    def peak(self, z, name='x'):
        ## SEE HELPERS
        pass

    def tile_and_print(self, input, tiles_height, tiles_width, padding=1):  # taken from asgn2
        ## SEE HELPERS
        pass


class GAN_conditional(nn.Module):
    def __init__(self, criterion):
        super(GAN_conditional, self).__init__()
        self.generator = Adversary(z_size + 10, hs_g1, hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator(xout_size + 10, hs_d1, hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train()
        _y = SANKETNET.hot_helper(np.random.rand(batch_size) // .10292, labels_override=10)[0]
        self.seed = torch.cat(
            (torch.randn(batch_size, z_size), torch.tensor(_y)),
            axis=-1
        ).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            weight_decay=w_g,
            betas=(beta1_g, beta2_g)
        )
        self.optim_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            weight_decay=w_d,
            betas=(beta1_d, beta2_d)
        )

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth):
        score = score.to(device)
        truth = truth.to(device)
        # truth = y, score = y_hat
        return self.criterion(score, truth)  # takes mean reduction

    # def batches_loop(self):
    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        # count_correct = 0
        for x, y in fmnist_loader:
            batch_count += 1
            # data_rinse
            x = x.squeeze()
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            y = torch.tensor(SANKETNET.hot_helper(y.tolist(), labels_override=10)[0])
            y = y.to(device)
            x = torch.cat((x, y), axis=-1)

            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)
            z = torch.cat((z, y), axis=-1)

            g = torch.cat((self.generator(z), y), axis=-1)

            d_g = self.discriminator(g.detach())
            y_dg = torch.zeros(batch_size, 1)
            y_dg = y_dg.to(device)
            loss_dg = self.loss(d_g, y_dg)

            # discriminator forward w/ real
            d_x = self.discriminator(x)
            y_dx = torch.ones(batch_size, 1)
            y_dx = y_dx.to(device)
            # d = d_g + d_x

            # discriminator loss, backward, step
            loss_dx = self.loss(d_x, y_dx)
            loss_dt = loss_dx + loss_dg
            loss_total_d += loss_dt.item()
            optim_d.zero_grad()
            loss_dt.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), c_d)
            optim_d.step()

            if batch_count % rd == 0:
                for i in range(rg):
                    # generator forward
                    z = torch.randn(batch_size, z_size)  # rand latent
                    z = z.to(device)
                    z = torch.cat((z, y), axis=-1)

                    g = torch.cat((self.generator(z), y), axis=-1)

                    y_g = self.discriminator(g)  # fake score
                    y_gt = torch.ones(batch_size, 1)  # target
                    y_gt = y_gt.to(device)
                    y_g = y_g.to(device)

                    # generator loss, backward, step
                    loss_g = self.loss(y_g, y_gt)
                    optim_g.zero_grad()
                    loss_total_g += loss_g.item()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), c_g)
                    optim_g.step()
                    # steps the generators weights ....

            if batch_count % 200 == 0:
                print(batch_count,
                      f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

        for i, each in enumerate(self.generator.parameters()):
            print('generator weight norms', torch.norm(each)) if i % 2 == 0 else None
        for i, each in enumerate(self.discriminator.parameters()):
            print('discriminator weight norms', torch.norm(each)) if i % 2 == 0 else None
        # self.peak(z, name='train')
        self.loss_totals_g.append(loss_total_g)
        self.loss_totals_d.append(loss_total_d)
        self.score_g.append(d_g.detach().mean().item())
        self.score_d.append(d_x.detach().mean().item())
        return

    def train_gan(self):
        count_epoch = 0
        for epoch in range(epochs):
            count_epoch += 1
            print('EPOCH:', count_epoch)
            self.batches_loop()
            self.peak(self.seed, name='ss')

        save_bin(f'gan', self)

    def peak(self, z, name='x'):
        ## SEE HELPER
        pass

    def tile_and_print(self, input, tiles_height, tiles_width, padding=1):  # taken from asgn2
        ## SEE HELPERS
        pass

    def visualize(self):
        batch_size = 30
        inv_normalize = transforms.Normalize(
            mean=[-0.1307 / 0.3081],
            std=[1 / 0.3081]
        )
        li = []
        for i in [[i] * 3 for i in range(10)]:
            li += i
        _y = SANKETNET.hot_helper(
            li, labels_override=10)[0]
        _x = torch.cat(
            (torch.randn(batch_size, 200), torch.tensor(_y)), axis=-1
        ).to(device)
        cGAN = torch.load('./Resources/p5/gan_1620868521.pt', map_location='cpu')
        g = cGAN(_x)
        g = g.reshape(batch_size, 1, 28, 28)
        g = inv_normalize(g)
        tiles = self.tile_and_print(g, 10, 3)
        tiles = tiles.permute(1, 2, 0)
        tiles = tiles.cpu().detach().numpy()
        tiles = tiles.squeeze()
        plt.figure(figsize=(20, 60))
        plt.imshow(tiles, interpolation='bilinear', cmap='gray')
        plt.savefig(f'./plots/cGAN_{int(time.time())}.png')
        plt.close()  # shows up as green?



def problem2(loss):
    gan = GAN(loss)
    gan.train_gan()
    plot_loss(gan)
    plot_scores(gan)
    return gan


def problem2b(loss):
    gan = GAN_MSE(loss)
    gan.train_gan()
    plot_loss(gan)
    # plot_scores(gan)
    return gan


def problem2c(loss):
    gan = GAN_Wass(loss)
    gan.train_gan()
    plot_loss(gan)
    plot_scores(gan)
    return gan


def problem2d(loss):
    gan = GAN_lsq(loss)
    gan.train_gan()
    plot_loss(gan)
    plot_scores(gan)
    return gan


def problem3_train():
    nn = FMnist_classifier()
    nn.train_classifier()
    return nn


def problem3_histo():
    li = []
    f = torch.load('./Resources/p3/class_1620793876.pt')
    g = torch.load('./Resources/p2/gan_1620801902.pt', map_location='cpu')
    for i in range(3000):
        z = torch.randn(1, z_size)
        p = g(z)
        y = f(p).argmax().item()
        li.append(y)

    class Shell():
        def __init__(self):
            self.histogram = li

    return Shell()


def problem4(loss):
    gan = GAN_unrolled(loss)
    gan.train_gan()
    plot_loss(gan)
    plot_scores(gan)
    return gan


def problem5(loss):
    gan = GAN_conditional(loss)
    gan.train_gan()
    plot_loss(gan)
    plot_scores(gan)
    return gan


def plot_loss(net):
    x = range(epochs)
    y2 = {'data': net.loss_totals_d, 'label': 'discriminator'}
    y1 = {'data': net.loss_totals_g, 'label': 'generator'}
    SANKETNET.Plot.curves(
        x,
        y1,
        y2,
        ind_label='epochs',
        dep_label='loss',
        title=f'Vanilla Gan Loss {int(time.time())}',
        yscale='log'
    )


def plot_scores(net):
    x = range(epochs)
    y2 = {'data': net.score_d, 'label': 'discriminator'}
    y1 = {'data': net.score_g, 'label': 'generator'}
    SANKETNET.Plot.curves(
        x,
        y1,
        y2,
        ind_label='epochs',
        dep_label='scores',
        title=f'Vanilla Gan Scores {int(time.time())}')


def plot_histogram(net):
    SANKETNET.Plot.histogram(
        (net.histogram, [i for i in range(10)], 'classes'),
        ind_label='count',
        dep_label='class',
        title=f'Class Distribution {int(time.time())}')


def WassLoss(d_x, d_gx, d=True):
    if d:
        l = (d_gx - d_x).mean()
    else:
        l = (-d_gx).mean()
    return l

def LeaseSquareLoss(d_x, d_gz, d=True):
    # note that it's switched for generator..
    if d:
        l = (((d_x - 1) ** 2 + d_gz) ** 2).mean()
    else:
        l = ((d_gz - 1) ** 2).mean()
    return l

loss_2a = nn.BCELoss()
loss_2b = nn.MSELoss()
loss_2c = WassLoss
loss_2d = LeaseSquareLoss
