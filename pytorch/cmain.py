from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt

import math

import models.cdcgan as cdcgan
import models.mlp as mlp
import json

def get_pipe_label(pipe):
   if pipe == 1:
      return [1,0,0]
   elif pipe == 2:
      return [0,1,0]
   elif pipe == 3:
      return [0,0,1]
   
def get_mario_playable(pipe):
    mario_chars_unique = sorted(list(["-","X", "}", "{", "<", ">", "[", "]", "Q", "S"]))
    int2char = dict(enumerate(mario_chars_unique))

    char2int = {ch: ii for ii, ch in int2char.items()}
    num_tiles = len(char2int)

    levels = []
    labels = []
    current_block = []
    with open(f"../db/big-<-{pipe}", 'r') as file:
        for line in file:
            # line = line.replace("[", "<")
            # line = line.replace("]", ">")
            line = line.rstrip('\n')
            if all(char in mario_chars_unique for char in line):
                ncoded_line = [char2int[x] for x in line]
                current_block.append(ncoded_line)
            else:
                if len(current_block) > 0:
                    current_block = np.array(current_block).reshape(14, 32)
                    levels.append(current_block)
                    current_block = []
                    labels.append(get_pipe_label(pipe))

    # levels = np.eye(num_tiles, dtype='uint8')[levels]
    # levels = levels.reshape(-1,14, 32, 10)
    return levels, labels

#Run with "python main.py"

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, default=32, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')

parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--problem', type=int, default=0, help='Level examples')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
 
map_size = 32

if opt.problem == 0:
    examplesJson = "example.json"
else:
    examplesJson= "sepEx/examplemario{}.json".format(opt.problem)
# X = np.array ( json.load(open(examplesJson)) )
pipe_1_playble, pipe_1_playble_labels = get_mario_playable(1)
# pipe_1_unplayble, pipe_1_unplayble_labels = get_mario_unplayable(1)
pipe_2_playble, pipe_2_playble_labels = get_mario_playable(2)
# pipe_2_unplayble, pipe_2_unplayble_labels = get_mario_unplayable(2)
pipe_3_playble, pipe_3_playble_labels = get_mario_playable(3)
# pipe_3_unplayble, pipe_3_unplayble_labels = get_mario_unplayable(3)
X = np.concatenate((pipe_1_playble, pipe_2_playble, pipe_3_playble))
y = np.concatenate((pipe_1_playble_labels, pipe_2_playble_labels, pipe_3_playble_labels))

# print(X)
print(X.shape)
print(y.shape)
z_dims = 10 #Numer different title types
y_dims = y.shape[1]

num_batches = X.shape[0] / opt.batchSize

print ("SHAPE ",X.shape) 
X_onehot = np.eye(z_dims, dtype='uint8')[X]

X_onehot = np.rollaxis(X_onehot, 3, 1)
print ("SHAPE ",X_onehot.shape)    #(173, 14, 28, 16)

X_train = np.zeros ( (X.shape[0], z_dims, map_size, map_size) )*2

X_train[:, 2, :, :] = 1.0  #Fill with empty space

#Pad part of level so its a square
X_train[:X.shape[0], :, :X.shape[1], :X.shape[2]] = X_onehot
y_train = y

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)

n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

netG = cdcgan.DCGAN_G(map_size, nz + y_dims, z_dims, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = cdcgan.DCGAN_D(map_size, nz, z_dims + y_dims, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, z_dims, map_size, map_size)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

def tiles2image(tiles):
    return plt.get_cmap('rainbow')(tiles/float(z_dims))

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1],shape[2]), dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img
    return image

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    print("Using ADAM")
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

gen_iterations = 0
for epoch in range(opt.niter):
    
    #! data_iter = iter(dataloader)

    length = len(X_train)
    perm = torch.randperm(length)

    X_train = X_train[perm]
    y_train = y_train[perm]

    i = 0
    while i < num_batches:#len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        # train the discriminator Diters times
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            Diters = 100
        else:
            Diters = opt.Diters
        j = 0
        while j < Diters and i < num_batches:#len(dataloader):
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

            data = X_train[i*opt.batchSize:(i+1)*opt.batchSize]
            label = y_train[i*opt.batchSize:(i+1)*opt.batchSize]
            i += 1

            real_cpu = torch.FloatTensor(data)
            real_label = torch.FloatTensor(label)

            if (False):
                #im = data.cpu().numpy()
                print(data.shape)
                real_cpu = combine_images( tiles2image( np.argmax(data, axis = 1) ) )
                print(real_cpu)
                plt.imsave('{0}/real_samples.png'.format(opt.experiment), real_cpu)
                exit()
           
            netD.zero_grad()
            #batch_size = num_samples #real_cpu.size(0)

            if opt.cuda:
                real_cpu = real_cpu.cuda()
                real_label = real_label.cuda()

            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)
            labelv = Variable(real_label)

            errD_real = netD(inputv, labelv)
            errD_real.backward(one)

            # train with fake
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise, volatile = True) # totally freeze netG
            fake = Variable(netG(noisev, labelv).data)
            inputv = fake
            errD_fake = netD(inputv, labelv)
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        netG.zero_grad()
        # in case our last batch was the tail batch of the dataloader,
        # make sure we feed a full batch of noise
        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, labelv)
        errG = netD(fake, labelv)
        errG.backward(one)
        optimizerG.step()
        gen_iterations += 1

        print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
            % (epoch, opt.niter, i, num_batches, gen_iterations,
            errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))
        if gen_iterations % 50 == 0:   #was 500

            fake = netG(Variable(fixed_noise, volatile=True), labelv)
            
            im = fake.data.cpu().numpy()
            #print('SHAPE fake',type(im), im.shape)
            #print('SUM ',np.sum( im, axis = 1) )

            im = combine_images( tiles2image( np.argmax( im, axis = 1) ) )

            plt.imsave('{0}/mario_fake_samples_{1}.png'.format(opt.experiment, gen_iterations), im)
            torch.save(netG.state_dict(), '{0}/netG_epoch_{1}_{2}_{3}.pth'.format(opt.experiment, gen_iterations, opt.problem, opt.nz))

    # do checkpointing
    #torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    #torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
