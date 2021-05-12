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

from torch.utils.tensorboard import SummaryWriter

Lambda = transforms.Lambda
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Problem 0 done by hand


# Problem 1 : Data Prep
# stamp = int(time.time())


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


# Params
paths = {
    1: "text/all_merged.txt",
    2: 'text/test_neg_merged.txt',
    3: 'text/test_pos_merged.txt',
    4: 'text/train_neg_merged.txt',
    5: "text/train_pos_merged.txt"
}
# batch_size = 64
# hidden_size = 64
# embed_size = 32
# learning_rate = .01
# epochs = 30  # iterations
# w = .003


# Make Network class module
# nn.embedding is a class table object thing with methods. like a wrapper on a tensor
class GRURnnet(nn.Module):
    def __init__(self, voc_size, embed_size, seq_len, hidden_size):  # input size = seq size
        super(GRURnnet,self).__init__()
        self.voc_size = voc_size
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size

        # Embed layer is what wraps the the table which wraps the words in feature vectors....row
        # dim is vocab size, col dim is embed dim, ie 32, of choosing....the weights in the layer
        # are trained and are the thing trainined. The input here is the batch of sentences as
        # indices...so (BATCH x SEQ_LEN)
        # output is tables of embeds....so (BATCH x SEQ_LEN x EMBED_DIM).
        # self.h0, defailts to 0
        self.embed = nn.Embedding(self.voc_size, self.embed_size)

        # GRU is like normal rnn, with extra gates (and weights for those gates). 3x the weights
        # for each moment. input here is the sequence...shape (BATCH x SEQ_LEN x EMBED_DIM).
        # ouput is both the last hidden state as well as the whole output vector across seq...so
        # shapes (BATCH x SEQ_LEN x HIDDEN_DIM) and (BATCH x HIDDEN_DIM)...things like
        # bi-directionality and extra GRU layers add dimensionality...
        self.gru = nn.GRU(input_size=self.embed_size, hidden_size=self.hidden_size,
                          batch_first=True)

        # Good 'ol Linear layer is what receives the GRU output, which is still the hidden size,
        # needs to bdownsized to the final output
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        #       assert x.shape == (batch_size,1,self.seq_len)
        x = x.squeeze()
        x = self.embed(x)
        x = self.gru(x)
        x = x[0][:, -1, :]
        x = self.linear(x)
        x = nn.Sigmoid()(x)
        return x.float()

    def init_hidden(self):
        self.h0 = torch.zeros(batch_size, self.hidden_size)  # unclear why cude?


class MLP_net(nn.Module):
    def __init__(self, seq_len, voc_size):
        super(MLP_net,self).__init__()
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
            torch.nn.utils.clip_grad_norm_(model.parameters(),3)
            optimizer.step()
        if batch_count % 50 == 0:
            print(batch_count, ' batches complete')
        # Accuracy Comp
        count_correct += (y == torch.round(y_hat)).sum()
    accuracy = (count_correct / (batch_count * batch_size))
    return loss_total, accuracy


def problem_1(network, train_loader, test_loader):
    network.to(device)
    # w = 0
    # if type(network) == MLP_net:
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=w)  # is
    # this just a
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


# Data
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


# PROBLEM 1 MAIN FUNCTIONS (UNCOMMENT)
# vocabulary = vocabulary_builder(strip_split(paths[1]))
# train_loader, test_loader, voc_size, input_size = data_prep()
# grurnnet = GRURnnet(voc_size, embed_size, input_size, hidden_size)
# # mlp_net = MLP_net(input_size, voc_size)
# # mlp = problem_1(mlp_net, train_loader, test_loader)
# rnn = problem_1(grurnnet, train_loader, test_loader)

# torch.randn
# q why does adam need a learning rate input...how does it work anyway? still sgd right? why need
#  decay still?
# q why doesn't mlp work here? Seems like it should....maybe would just get unweildy?

# Essentially, a gan is a competition between 2 networks, A and B. A is a generator,
# B is a discriminator, sometimes called critic. the basic flow is A 'generates' sample -> sends
# it to B -> B determines whether it's fake or real. This decision is compared against the truth
# and loss calculated. this loss backprops all the way to the generator, which then adjusts it's
# 'generation' to better fool the critic. Then, the critic takes in a real input and does the
# same thing. Only this time, the loss backprop does not go all the way to the generator. Or if
# it does, gradients are reset when the generator trains anyway. # q unclear here why not just
# reset. # q also unclear here why not feed them in at same time.
# the critic then adjusts it's weights. In a sense they are trying to optimize an objective
# function with the loss function. Use sigmoid for the discriminator output, and tanh for the
# generator output, which has a range of -1,1, which I'd imagine the inputs must also be
# normalized to. GANS training poses many challenges and questions...When is training done? What
# does convergence look like? What is a good outcome? How to deal with 'multimodal'
# distributions? (q Why not make the z multimodal? or why not have multiple adversarial
#  components to a single discriminator?) There are also problems such as modal collapse,
#  non-convergence, and vanishing gradients.
# Note on Unroll Gan and "replay:
# Note on Conditional GAN:

#
#  chatted with Jan about this already but hope some of these help others since its due soon (just personal things that helped me):
# Adjusting and dropping beta1 on Adam optimizer
# Increasing LR (1e-4 - 1e-5 range was where my optimal values were)
# Try LeakyReLU instead of plain ReLU
# Increasing hidden layer width for generator (increases capacity) or decreasing it for discriminator if discriminator keeps on converging to 0 loss
# Triple checking that loss function is correct (I used sum instead of mean in place of expectation by mistake - shouldn’t technically be an issue generally but might be in some cases)
# Using clamp_ instead of clamp because the first is in place (took me like a day to figure that out)
# Re initialising optimizer, discriminator, and generator for each value of C
# Hope that was helpful to someone
# Problem 2
# part 1, vanilla gan
# Hypers & Vars
# batch_size = 32
# xout_size = 28 * 28
#
# hidden_size_g = 800
# hidden_size_d = 1600
# z_size = 200
# learning_rate_g = .001
# learning_rate_d = .001
# epochs = 50  # iterations
# w = 0
# beta1 = .5
# beta2 = .9
# Dataset/load (standard Fashion mnist)
# normalize data when treating....
# hypers = {
#     'max_epochs': 10,
#     'batch_size': 32,
#     'g_lr': 1e-4,
#     'd_lr': 1e-4,
#     'b1': 0.5,
#     'b2': 0.999,
#
#     'z_dim': 100,
#     'x_dim': 2,
#     'latent_dim_G': 100,
#     'latent_dim_D': 100,
# }
def transform(img):
    img = transforms.ToTensor()(img)  # 0 to 1 tensor
    img = transforms.Normalize(mean=0.1307, std=0.3081)(img)
    return img

fmnist = torchvision.datasets.FashionMNIST(
    root="./", train=True,
    transform=transform, download=True)

fmnist_loader = torch.utils.data.DataLoader(
    dataset=fmnist,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)

fmnist_test = torchvision.datasets.FashionMNIST(
    root="./",
    transform=transform, download=True)

fmnist_test_loader = torch.utils.data.DataLoader(
    dataset=fmnist_test,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True)

# generator module
# 3 hidden linear layers with ReLU activation and Tanh output activation

class Adversary(nn.Module):
    def __init__(self, z_size, hs1, hs2, hs3, xout_size):
        super(Adversary,self).__init__()
        self.lin1 = nn.Linear(z_size, hs1)
        self.lin2 = nn.Linear(hs1,hs2)
        self.lin3 = nn.Linear(hs2, hs3)
        self.output = nn.Linear(hs3, xout_size)
        self.a1 = nn.LeakyReLU(.2) # CONSIDER LEAKY RELU
        self.a2 = nn.Tanh()
        pass
    
    def forward(self,z):
        z = self.lin1(z)
        z = self.a1(z)
        z = self.lin2(z)
        z = self.a1(z)
        z = self.lin3(z)
        z = self.a1(z)
        z = self.output(z)
        z = self.a2(z)
        return z # generated output
        
class Discriminator(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(input_size, hs1)
        self.lin2 = nn.Linear(hs1, hs2)
        self.lin3 = nn.Linear(hs2, hs3)
        self.output = nn.Linear(hs3, 1)
        self.a1 = nn.LeakyReLU(.2)  # CONSIDER LEAKY RELU
        self.a2 = nn.Sigmoid()

    def forward(self,x):
        x = self.lin1(x)
        x = self.a1(x)
        x = self.lin2(x)
        x = self.a1(x)
        x = self.lin3(x)
        x = self.a1(x)
        x = self.output(x)
        x = self.a2(x)
        return x # real/fake score

    def peak_weights(self):
        for each in self.parameters():
            print()


class Discriminator_Wass(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3):
        super(Discriminator_Wass, self).__init__()
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

class Discriminator_classifier(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3):
        super(Discriminator_classifier, self).__init__()
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
        # x = self.a2(x)
        return x  # real/fake score


# discriminator modules
# 3 hidden linears with Relu, output sigmoud....use BCE loss for optimizers (will have 2 separate
# optimizer objects)

# overall network system module....will have training funcitons embedded, like sanketnet.
class GAN(nn.Module):
    def __init__(self, criterion):
        super(GAN, self).__init__()
        #COmponents
        self.generator = Adversary(z_size, hs_g1,hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator(xout_size, hs_d1,hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train() # NEcessary? maybe not
        # Cache and Met rics
        # Do we want to cache every output of the model? or just random seed sample it every so
        # epochs...just to check in on it....essentially "sample" from the dist we're modeling..
        # I think just do both?
        #self.replay # if you wanted
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr = learning_rate_g,
                                   weight_decay=w_g, betas= (beta1_g, beta2_g))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_d,
                                   weight_decay=w_d, betas= (beta1_d, beta2_d))

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth):
        score = score.to(device)
        truth = truth.to(device)
        # truth = y, score = y_hat
        return self.criterion(score, truth) # takes mean reduction

    # def batches_loop(self):
    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        # count_correct = 0
        for x, _ in fmnist_loader:
            batch_count += 1
            # data_rinse
            x = x.squeeze() # move data treatment to data funciton
            x = x.reshape(batch_size,xout_size)
            x = x.to(device)
            # The sampling of ze (shape z-size by something)
            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)
            # discriminator forward w/ fake
            d_g = self.discriminator(self.generator(z).detach())
            y_dg = torch.zeros(batch_size, 1)
            y_dg = y_dg.to(device)
            loss_dg = self.loss(d_g, y_dg)
            # loss_total_d += loss_dg.item()
            # loss_dg.backward()

            # discriminator takes the same output and feeds forward on it's network (again) (for
            # gradient purposes) can potentially detach here....try detach first and then
            # without...in interest of time...

            # discriminator forward w/ real
            d_x = self.discriminator(x)
            y_dx = torch.ones(batch_size,1)
            y_dx = y_dx.to(device)
            # d = d_g + d_x

            # discriminator loss, backward, step
            loss_dx = self.loss(d_x, y_dx )
            loss_dt = loss_dx + loss_dg
            loss_total_d += loss_dt.item()
            optim_d.zero_grad()
            loss_dt.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), c_d)
            optim_d.step()

            # Feeding of z into generator. for some reason don't need to seed this...what would
            # happend if we did?
            # generator's output is already normalized, goes into the discriminator forward
            # discriminator's output goes into loss fn, along with a vector of 1's
            # clear the gradient
            # loss fn backprops all the way back to generator, store loss

            if batch_count % rd == 0:
                for i in range(rg):
                #generator forward
                    z = torch.randn(batch_size, z_size)  # rand latent
                    z = z.to(device)
                    g = self.generator(z)
                    y_g = self.discriminator(g) # fake score
                    y_gt = torch.ones(batch_size, 1) # target
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
                print(batch_count, f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

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
        # takes a sample from the data-space (input) via data loader, and feeds it forward through
        # the discriminator
        # take a loss of that, and backprop. and step (
        # disc params specified in the optimizer config)
        # step the discriminator weights

        # Note: Why only 1 optimizer step? because the derivitive of a sum is the sum of it's
        # deriviates. So you can either combine the results and backprop that, or you can
        # backprop them seperately, in which case the gradients will be added together! and so
        # yuo only do 1 step. actually cannot add bc we need different loss directions...lol

        # add both losses to net caches (before return). also add the real/fake forward ouputs (
        # scores) to the
        # cache...

        # every epoch, take a "peak". In a consistently seeded sample of z, do a forward call on
        # the generator, and visualize the results. this is as simple as just saving down the
        # file, overwriting everytim....can just pull in the tile and view fn from asgn 2.

    def train_gan(self):
        count_epoch = 0
        for epoch in range(epochs):
            count_epoch += 1
            print('EPOCH:', count_epoch)
            self.batches_loop()
            self.peak(self.seed, name='ss')

        save_bin(f'gan', self)

    def peak(self, z, name = 'x'):
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
        plt.close() # shows up as green?

    def tile_and_print(self,input, tiles_height, tiles_width, padding=1): # taken from asgn2
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



class GAN_MSE(nn.Module):
    def __init__(self, criterion):
        super(GAN_MSE, self).__init__()
        # COmponents
        self.generator = Adversary(z_size, hs_g1, hs_g2, hs_g3, xout_size)
        self.criterion = criterion
        self.to(device)
        self.train()  # NEcessary? maybe not
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_d = []
        self.score_g = []
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr=learning_rate_g,
                                        weight_decay=w_g, betas=(beta1_g, beta2_g))

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth):
        score = score.to(device)
        truth = truth.to(device)
        # truth = y, score = y_hat
        return self.criterion(score, truth)  # takes mean reduction

    def batches_loop(self):
        optim_g = self.optim_g
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        for x, _ in fmnist_loader:
            batch_count += 1

            # data_rinse
            x = x.squeeze()  # move data treatment to data funciton
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)

            y_g = self.generator(z)
            y_g = y_g.to(device)

            # generator loss, backward, step
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

# overall network system module....will have training funcitons embedded, like sanketnet.
class GAN_Wass(nn.Module):
    def __init__(self, criterion):
        super(GAN_Wass, self).__init__()
        # Components
        self.generator = Adversary(z_size, hs_g1,hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator_Wass(xout_size, hs_d1,hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train() # NEcessary? maybe not
        # Cache and Met rics
        # Self.replay # if you wanted
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr = learning_rate_g,
                                   weight_decay=w_g, betas= (beta1_g, beta2_g))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_d,
                                   weight_decay=w_d, betas= (beta1_d, beta2_d))

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth, d):
        if isinstance(score, int) or isinstance(score, float):
            score = torch.tensor(score)
        if isinstance(truth, int) or isinstance(truth, float):
            truth = torch.tensor(truth)
        score = score.to(device)
        truth = truth.to(device)
        # truth = y, score = y_hat
        return self.criterion(score, truth, d) # takes mean reduction

    # def batches_loop(self):
    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        # count_correct = 0
        for x, _ in fmnist_loader:
            batch_count += 1

            # data_rinse
            x = x.squeeze() # move data treatment to data funciton
            x = x.reshape(batch_size,xout_size)
            x = x.to(device)
            # The sampling of ze (shape z-size by something)
            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)
            # generator forward
                # steps the generators weights ....
            d_g = self.discriminator(self.generator(z).detach())

            # Q is it supposed to sample newly each time?

            # discriminator forward w/ fake
            # y_dg = torch.zeros(batch_size, 1)
            # y_dg = y_dg.to(device)
            # loss_dg = self.loss(d_g, y_dg)
            # loss_total_d += loss_dg.item()
            # loss_dg.backward()

            # discriminator forward w/ real
            d_x = self.discriminator(x)
            # y_dx = torch.ones(batch_size,1)
            # y_dx = y_dx.to(device)
            # d = d_g + d_x

            # discriminator loss, backward, step
            loss_dx = self.loss(d_x, d_g, d = True)
            # loss_dt = loss_dx # +loss_dg
            loss_total_d += loss_dx.item()
            optim_d.zero_grad()
            loss_dx.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), c_d)
            optim_d.step()
            for p, each in enumerate(self.discriminator.parameters()):
                if p % 2 == 0:
                    # with torch.no_grad():
                    each.data.clamp_( -c_w, c_w)

            if batch_count % rd == 0:
                for i in range(rg):
                    z = torch.randn(batch_size, z_size)  # rand latent
                    z = z.to(device)

                    # generator forward
                    g = self.generator(z)
                    y_g = self.discriminator(g) # fake score
                    # y_gt = torch.ones(batch_size, 1) # target
                    # y_gt = y_gt.to(device)
                    y_g = y_g.to(device)

                    # generator loss, backward, step
                    loss_g = self.loss(0, y_g, d = False)
                    optim_g.zero_grad()
                    loss_total_g += loss_g.item()
                    loss_g.backward()
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), c_g)
                    optim_g.step()

            if batch_count % 200 == 0:
                print(batch_count, f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

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

    def peak(self, z, name = 'x'):
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
        plt.close() # shows up as green?

    def tile_and_print(self,input, tiles_height, tiles_width, padding=1): # taken from asgn2
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
        # Components
        self.generator = Adversary(z_size, hs_g1, hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator(xout_size, hs_d1, hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train()  # NEcessary? maybe not
        # Cache and Met rics
        # Self.replay # if you wanted
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
        # truth = y, score = y_hat
        return self.criterion(score, truth, d)  # takes mean reduction

    # def batches_loop(self):
    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        # count_correct = 0
        for x, _ in fmnist_loader:
            batch_count += 1

            # data_rinse
            x = x.squeeze()  # move data treatment to data funciton
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            # The sampling of ze (shape z-size by something)
            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)

            d_g = self.discriminator(self.generator(z).detach())
            # y_dg = torch.zeros(batch_size, 1)
            # y_dg = y_dg.to(device)
            # loss_dg = self.loss(d_g, y_dg)
            # loss_total_d += loss_dg.item()
            # loss_dg.backward()

            # discriminator forward w/ real
            d_x = self.discriminator(x)
            # y_dx = torch.ones(batch_size,1)
            # y_dx = y_dx.to(device)
            # d = d_g + d_x

            # discriminator loss, backward, step
            loss_dx = self.loss(d_x, d_g, d=True)
            loss_dt = loss_dx  # +loss_dg
            loss_total_d += loss_dt.item()
            optim_d.zero_grad()
            loss_dt.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), c_d)
            optim_d.step()

            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)
            # generator forward
            g = self.generator(z)
            y_g = self.discriminator(g)  # fake score
            # y_gt = torch.ones(batch_size, 1) # target
            # y_gt = y_gt.to(device)
            y_g = y_g.to(device)

            # generator loss, backward, step
            loss_g = self.loss(0,y_g, d = False)
            optim_g.zero_grad()
            loss_total_g += loss_g.item()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), c_g)
            optim_g.step()

            # steps the generators weights ....

            # discriminator forward w/ fake


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


class FMnist_classifier(nn.Module):
    def __init__(self):
        super(FMnist_classifier, self).__init__()
        # Components
        self.discriminator = Discriminator_classifier(xout_size, hs_d1,hs_d2, hs_d3)
        self.criterion = nn.CrossEntropyLoss()
        self.to(device)
        self.loss_totals = []
        self.loss_totals_test = []
        self.accuracy = []
        self.accuracy_test = []
        self.histogram = []
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_d)
                                   # weight_decay=w_d, betas= (beta1_d, beta2_d))

    def forward(self, input):
        return self.discriminator(input)

    def loss(self, y_hat, y, d):
        # if isinstance(score, int) or isinstance(score, float):
        #     score = torch.tensor(score)
        # if isinstance(truth, int) or isinstance(truth, float):
        #     truth = torch.tensor(truth)
        y_hat = y_hat.to(device)
        y = y.to(device)
        # truth = y, score = y_hat
        return self.criterion(y_hat, y, d) # takes mean reduction


    def batches_loop(self, loader, is_val=False):
        model = self.forward
        optimizer = self.optimizer
        criterion = self.criterion
        batch_count = 0
        loss_total = 0
        count_correct = 0
        for x, y in loader:
            batch_count += 1
            x = x.squeeze() # move data treatment to data funciton
            x = x.reshape(batch_size,xout_size)
            x = x.to(device)
            y = y.to(device)
            # assert x.shape[0] == hypes["BATCH"], x.shape
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
                print(batch_count,f'batches complete')

        # evaluate, record
            y_hat_arg = y_hat.argmax(dim=-1)
            self.histogram += y_hat_arg.tolist()
            # print(1, y_hat_arg)
            # print(2 ,y)
            # print(3, count_correct)
            # print(4, (y == y_hat_arg).sum())
            count_correct += (y == y_hat_arg).sum()
            # print(5, count_correct, (batch_count * batch_size))

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
            self.batches_loop(fmnist_test_loader, is_val= True)
        metrics = (self.loss_totals, self.loss_totals_test, self.accuracy,self.accuracy_test)
        save_bin(f'class', network)
        save_bin(f'class_metrics', metrics)


# overall network system module....will have training funcitons embedded, like sanketnet.
class GAN_unrolled(nn.Module):
    def __init__(self, criterion):
        super(GAN_unrolled, self).__init__()
        #COmponents
        self.generator = Adversary(z_size, hs_g1,hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator(xout_size, hs_d1,hs_d2, hs_d3)
        self.criterion = criterion
        self.to(device)
        self.train() # NEcessary? maybe not
        # Cache and Met rics
        # Do we want to cache every output of the model? or just random seed sample it every so
        # epochs...just to check in on it....essentially "sample" from the dist we're modeling..
        # I think just do both?
        #self.replay # if you wanted
        self.seed = torch.randn(batch_size, z_size).to(device)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr = learning_rate_g,
                                   weight_decay=w_g, betas= (beta1_g, beta2_g))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_d,
                                   weight_decay=w_d, betas= (beta1_d, beta2_d))

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth):
        score = score.to(device)
        truth = truth.to(device)
        # truth = y, score = y_hat
        return self.criterion(score, truth) # takes mean reduction

    # def batches_loop(self):
    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        # count_correct = 0
        length = len(fmnist_loader)
        fmnist_iter = iter(fmnist_loader)
        for x, _ in fmnist_iter:
            batch_count += 1
            # data_rinse
            x = x.squeeze() # move data treatment to data funciton
            x = x.reshape(batch_size, xout_size)
            x = x.to(device)
            z = torch.randn(batch_size, z_size)  # rand latent
            z = z.to(device)

            d_g = self.discriminator(self.generator(z).detach())
            y_dg = torch.zeros(batch_size, 1)
            y_dg = y_dg.to(device)
            loss_dg = self.loss(d_g, y_dg)
            # loss_total_d += loss_dg.item()
            # loss_dg.backward()

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

            # q would it matter if the gen code was first? essentially if during the discrim loop, that the
            #  gen inpput has to ba new input from same space, or prior one...
            if batch_count % rd == 0:
                assert rolls > 0, 'unrolled gan needs rolls'
                # while batch_count < length - rolls:
                for k in range(rolls):
                    batch_count += 1
                    try:
                        _x, _ = next(fmnist_iter)
                    except StopIteration:
                        "Iteration Finished, breaking"
                        break
                    _x = _x.squeeze()  # move data treatment to data funciton
                    _x = _x.reshape(batch_size, xout_size)
                    _x = _x.to(device)
                    unrolled_discriminator = type(self.discriminator)(xout_size, hs_d1,hs_d2, hs_d3)  # get a new instance
                    unrolled_discriminator.load_state_dict(
                        self.discriminator.state_dict())  # copy weights and stuff
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
                    torch.nn.utils.clip_grad_norm_(unrolled_discriminator.parameters(), c_d)
                    optim_d.step()

                for i in range(rg):
                #generator forward
                    z = torch.randn(batch_size, z_size)  # rand latent
                    z = z.to(device)
                    g = self.generator(z)
                    y_g = unrolled_discriminator(g) # fake score
                    y_gt = torch.ones(batch_size, 1) # target
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
                print(batch_count, f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

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

    def peak(self, z, name = 'x'):
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
        plt.close() # shows up as green?

    def tile_and_print(self,input, tiles_height, tiles_width, padding=1): # taken from asgn2
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
    return li

def problem4(loss):
    gan = GAN_unrolled(loss)
    gan.train_gan()
    plot_loss(gan)
    plot_scores(gan)
    return gan

def plot_loss(net):
    x = range(epochs)
    y2 = {'data':net.loss_totals_d, 'label':'discriminator'}
    y1 = {'data':net.loss_totals_g,'label':'generator'}
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
        dep_label='loss',
        title=f'Vanilla Gan Scores {int(time.time())}')

def plot_histogram(net):
    SANKETNET.Plot.histogram(
        (net.histogram, [i for i in range(10)], 'classes'),
        ind_label='count',
        dep_label='class',
        title=f'Class Distribution {int(time.time())}')

def WassLoss(d_x, d_gx, d = True):
    if d:
        l = (d_gx - d_x).mean()
    else:
        l = (-d_gx).mean()
    return l
# print(c)

def LeaseSquareLoss(d_x, d_gz, d = True):
    # note that it's switched for generator..
    if d:
        l = ( ((d_x - 1)**2 + d_gz)**2 ).mean()
    else:
        l = ((d_gz - 1)**2).mean()
    return l

loss_2a = nn.BCELoss()
loss_2b = nn.MSELoss()
loss_2c = WassLoss
loss_2d = LeaseSquareLoss
# # loss_2b =
# if __name__ == "__main__":
#     # pass
#     problem2(loss_2a)
# training loop code...

# todo show generated samples from beginning of training, intermediate stage of training and
#  after 'convergence'

# todo plot loss curces.
# hook class
# def fmap_hook(module, input, output):
#     num_channels = output.shape[1]
#     assert num_channels > 5, print(output.shape)
#     num_channels = list(range(num_channels))
#     inds = [random.choice(num_channels) for _ in range(5)]
#     ft_maps_to_add = [output[0, i, ...] for i in inds]
#     assert len(ft_maps_to_add[0].shape) == 2
#     ft_maps_to_add = [transforms.ToPILImage()(o) for o in ft_maps_to_add]
#     # shape 1,c,h,w
#     # should output 5
#     for i in ft_maps_to_add:
#         ft_maps.append(i)
#     return


#
# for p in pepper_net.named_modules():
#     mod = p[1]
#     if p[0] in layers:
#         mod.register_forward_hook(fmap_hook)
#
# for i, each in enumerate(m.discriminator.parameters()):
#
# torch.norm
# # torch.cltorch.norm(each)) if i % 2 == 0 else None
#
#
# def gradient_clipper(model: nn.Module, val: float) -> nn.Module:
#     for parameter in model.parameters():
#         parameter.register_hook(lambda grad: grad.clamp_(-val, val))
#
#     return model
# torch.cli
#
# if clip_coef < 1:
#     for p in parameters:
#         p.grad.detach().mul_(clip_coef.to(p.grad.device))
# return total_norm
#
# torch.clamp()
#
# c_w = .1
# for p, each in enumerate(m.generator.parameters()):
#     if p % 2 == 0:
#
#         w_norm = torch.norm(each, p=1)
#         clip_coef = c_w / w_norm + 1e-6
#         m = c_w / each.size
#         if clip_coef < 1:
#             torch.clamp(m.generator.parameters()[p], m)
#

