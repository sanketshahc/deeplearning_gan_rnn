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
    if type(f_object) == torch.Tensor:
        name = f'{name}_{stamp}.pt'
        torch.save(f_object, f"pickled_binaries/{name}")
    else:
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

# generator module
# 3 hidden linear layers with ReLU activation and Tanh output activation

class Adversary(nn.Module):
    def __init__(self, z_size, hs1, hs2, hs3, xout_size):
        super(Adversary,self).__init__()
        self.lin1 = nn.Linear(z_size, hs1)
        self.lin2 = nn.Linear(hs1,hs2)
        self.lin3 = nn.Linear(hs2, hs3)
        self.output = nn.Linear(hs3, xout_size)
        self.a1 = nn.LeakyReLU() # CONSIDER LEAKY RELU
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
        
nn.LeakyReLU()
class Discriminator(nn.Module):
    def __init__(self, input_size, hs1, hs2, hs3):
        super(Discriminator, self).__init__()
        self.lin1 = nn.Linear(input_size, hs1)
        self.lin2 = nn.Linear(hs1, hs2)
        self.lin3 = nn.Linear(hs2, hs3)
        self.output = nn.Linear(hs3, 1)
        self.a1 = nn.LeakyReLU()  # CONSIDER LEAKY RELU
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


# discriminator modules
# 3 hidden linears with Relu, output sigmoud....use BCE loss for optimizers (will have 2 separate
# optimizer objects)

# overall network system module....will have training funcitons embedded, like sanketnet.
class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        #COmponents
        self.generator = Adversary(z_size, hs_g1,hs_g2, hs_g3, xout_size)
        self.discriminator = Discriminator(xout_size, hs_d1,hs_d2, hs_d3)
        self.to(device)
        self.train() # NEcessary? maybe not
        # Cache and Met rics
        # Do we want to cache every output of the model? or just random seed sample it every so
        # epochs...just to check in on it....essentially "sample" from the dist we're modeling..
        # I think just do both?
        #self.replay # if you wanted
        self.seed = torch.randn(batch_size, z_size)
        self.loss_totals_g = []
        self.loss_totals_d = []
        self.score_g = []
        self.score_d = []
        self.optim_g = torch.optim.Adam(self.generator.parameters(), lr = learning_rate_g,
                                   weight_decay=w, betas= (beta1, beta2))
        self.optim_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate_d,
                                   weight_decay=w, betas= (beta1, beta2))

    def forward(self, input):
        return self.generator(input)

    def loss(self, score, truth):
        score = score.to(device)
        truth = truth.to(device)
        # truth = y, score = y_hat
        return nn.BCELoss()(score, truth) # takes mean reduction

    # def batches_loop(self):
    def batches_loop(self):
        optim_g = self.optim_g
        optim_d = self.optim_d
        batch_count = 0
        loss_total_g = 0
        loss_total_d = 0
        # count_correct = 0
        for x, _ in fmnist_loader:
            x = x.squeeze() # move data treatment to data funciton
            x = x.reshape(batch_size,xout_size)
            batch_count += 1
            x = x.to(device)
            # The sampling of ze (shape z-size by something)
            z = torch.randn(batch_size, z_size) # rand latent
            z = z.to(device)

            # Feeding of z into generator. for some reason don't need to seed this...what would
            # happend if we did?
            self.peak(z, name='train') if batch_count % 1800 == 0 else None
            g = self.generator(z)
            y_g = self.discriminator(g) # fake score
            # generator's output is already normalized, goes into the discriminator forward
            y_gt = torch.ones(batch_size, 1) # target
            # y_gt= y_gt.float()
            y_gt = y_gt.to(device)
            y_g = y_g.to(device)
            # discriminator's output goes into loss fn, along with a vector of 1's
            loss_g = self.loss(y_g, y_gt)
            # clear the gradient
            optim_g.zero_grad()
            # loss fn backprops all the way back to generator, store loss
            loss_total_g += loss_g.item()
            loss_g.backward()
            # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 3)
            # steps the generators weights ....
            # clip grad?
            # torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 3)
            optim_g.step()

            d_g = self.discriminator(g.detach())
            y_dg = torch.zeros(batch_size, 1)
            y_dg = y_dg.to(device)
            loss_dg = self.loss(d_g, y_dg)
            loss_total_d += loss_dg.item()
            optim_d.zero_grad()
            loss_dg.backward()

            # discriminator takes the same output and feeds forward on it's network (again) (for
            # gradient purposes) can potentially detach here....try detach first and then
            # without...in interest of time...
            d_x = self.discriminator(x)
            y_dx = torch.ones(batch_size,1)
            y_dx = y_dx.to(device)
            loss_dx = self.loss(d_x, y_dx )
            # d = d_g + d_x
            loss_total_d += loss_dx.item()
            loss_dx.backward()
            # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 3)
            optim_d.step()
            if batch_count % 100 == 0:
                print(batch_count, f'batches complete, loss_g: {loss_total_g}, loss_d: {loss_total_d}')

        self.loss_totals_g.append(loss_total_g)
        self.loss_totals_d.append(loss_total_d)
        self.score_g.append(y_dg.detach().mean().item())
        self.score_d.append(y_dx.detach().mean().item())
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
        tiles = self.tile_and_print(g, 4, 8)
        tiles = tiles.permute(1, 2, 0)
        tiles = tiles.detach().numpy()
        plt.figure(figsize=(80, 40))
        plt.imshow(tiles, interpolation='bilinear')
        plt.savefig(f'./plots/peak_{name}_{int(time.time())}.png') # shows up as green?

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
        if padding:
            w = torch.cat((w, torch.ones((co, ci, p, wi), device=device)), dim=-2)
            w = torch.cat((w, torch.ones((co, ci, he + p, p), device=device)), dim=-1)
            co, ci, he, wi = w.shape
        w = w.permute(1, 2, 3, 0)
        w = w.reshape(ci, he, wi, tiles_height, tiles_width)
        w = w.permute(0, 3, 1, 4, 2)
        w = w.reshape(ci, he * tiles_height, tiles_width * wi)
        return w

def problem2():
    gan = GAN()
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
        title=f'Vanilla Gan Loss {int(time.time())}')


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

problem2()
# training loop code...

# todo show generated samples from beginning of training, intermediate stage of training and
#  after 'convergence'

# todo plot loss curces.

