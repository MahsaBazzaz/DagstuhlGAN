import os
import pdb
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
from datetime import datetime
import subprocess
import sys
import json
import numpy
import models.dcgan as dcgan
import matplotlib.pyplot as plt
import math
import random

def map_output_to_symbols(integers):
    mario_chars_unique = sorted(list(["-","X", "}", "{", "<", ">", "[", "]", "Q", "S"])) #
    int2char = dict(enumerate(mario_chars_unique))
    return [[int2char[i.item()] for i in row] for row in integers]


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(sys.argv) ==1:
        modelToLoad = "DCGAN_epoch_2250_0_32.pth"
    else:
        modelToLoad = sys.argv[1]
    if len(sys.argv) >=3:
        nz = int(sys.argv[2])
    else:
        nz = 32

    batch_size = 20
    #nz = 10 #Dimensionality of latent vector

    imageSize = 32
    ngf = 64
    ngpu = 1
    n_extra_layers = 0
    z_dims = 10 #number different titles

    generator = dcgan.DCGAN_G(imageSize, nz, z_dims, ngf, ngpu, n_extra_layers)
    generator.load_state_dict(torch.load(modelToLoad, map_location=lambda storage, loc: storage))

    lv = torch.randn(batch_size, 32, 1, 1, device=device)
    latent_vector = torch.FloatTensor( lv ).view(batch_size, nz, 1, 1) 

    levels = generator(Variable(latent_vector, volatile=True))

    #levels.data = levels.data[:,:,:14,:28] #Cut of rest to fit the 14x28 tile dimensions

    level = levels.data.cpu().numpy()
    level = level[:,:,:14,:32] #Cut of rest to fit the 14x28 tile dimensions
    level = numpy.argmax( level, axis = 1)

    directory = './out/' + "mario" + "_DCGAN_" + datetime.now().strftime("%Y%m%d%H%M")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully.")
    else:
        print(f"Directory '{directory}' already exists.")
        
    for i in range(batch_size):
        output = level[i]
        result = map_output_to_symbols(output)
        result_string = ""
        for row in result:
            for item in row:
                result_string += item
            result_string += "\n"
            
        with open(directory + "/" + str(i) + ".lvl", "w") as file:
            # Write text to the file
            file.write(result_string)

        try:
            reach_move = "platform"
            script_path = '../../sturgeon/level2repath.py'
            arguments = ['--outfile', directory + "/" + str(i) + ".path.lvl",'--textfile', directory + "/" + str(i) + ".lvl",'--reach-move', reach_move]
            command = ['python', script_path] + arguments
            print(command)
            result = subprocess.run(command, check=True)
            if os.path.exists(directory + "/" + str(i) + ".path.lvl"):
                print("Path exists. Level Playble.")
                vis_path = directory + "/" + str(i) + ".path.lvl"
            else:
                print("Path does not exist. Level Unplayble.")
                vis_path = directory + "/" + str(i) + ".lvl"
        except subprocess.CalledProcessError:
            print("Path does not exist. Level Unplayble.")
            vis_path = directory + "/" + str(i) + ".lvl"

        
        script_path = '../../level2image/level2image.py'
        arguments = [vis_path,
                    '--fmt', 'png']
        command = ['python', script_path] + arguments
        subprocess.run(command, check=True)
