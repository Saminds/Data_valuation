#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import torch
from torchvision import datasets, transforms
from src_opt.utils.sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, cifar_iid, cifar_noniid, \
    FashionMnist_noniid, cifar_longtail
from src_opt.utils.options import args_parser
import ssl
import random
import numpy as np
import math

def get_noiseword():
    NoiseWord = ["0_NonIID", "1_LongTail", "2_LabelNoise", "3_LabelNoise2", "4_DataNoise", "5_GradientNoise", "6_RandomAttack", "7_ReverseGradient", "8_ConstantAttack", "9_New", "10_ScalingNoise"]
    return NoiseWord

def get_datasetserver(args):
    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_dataset_all = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)
        train_dataset, test_dataset = torch.utils.data.random_split(test_dataset_all, [args.sz, 10000-args.sz])
        return train_dataset, test_dataset

    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])
        test_dataset_all = datasets.FashionMNIST(data_dir, train=False, download=True,
                                            transform=apply_transform)
        train_dataset, test_dataset = torch.utils.data.random_split(test_dataset_all, [args.sz, 10000 - args.sz])
        return train_dataset, test_dataset


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    ssl._create_default_https_context = ssl._create_unverified_context

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=apply_transform)

        test_dataset_all = datasets.CIFAR10(data_dir, train=False, download=True,
                                            transform=apply_transform)
        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [8000, 2000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Long-tailed
            if args.noise == 1:
                user_groups = cifar_longtail(train_dataset,  args.num_users, args.noiselevel)
            elif args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose equal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist':
        data_dir = '../data/mnist/'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset_all = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [8000, 2000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample 0_NonIID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'fmnist':
        data_dir = '../data/fmnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                              transform=apply_transform)

        test_dataset_all = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        test_dataset, valid_dataset = torch.utils.data.random_split(test_dataset_all, [8000, 2000])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Long-tailed
            if args.noise == 1:
                user_groups = cifar_longtail(train_dataset,  args.num_users, args.noiselevel)
            # Sample 0_NonIID user data from Mnist
            elif args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = FashionMnist_noniid(train_dataset, args.num_users)

        # label noise # label noise by swapping even labels for their next odd - both valid_dataset and test_dataset includes odd labels
        # train_dataset.train_labels[train_dataset.train_labels == 0] = 1
        if args.noise == 2:
            train_dataset.targets = torch.tensor(train_dataset.targets) # convert label to integer to facilitate manipulation
            for i in range(0, 9, 2):
                train_dataset.targets[train_dataset.targets == i] = i + 1 #mislabling even labbels and change them to odd labels
                indices = []
            for i in range(len(valid_dataset)):# create new valid_dataset that includes all odd labels and put all of them in indices. then filter valid_dataset using indices
                data, label = valid_dataset[i]
                if label % 2 != 0:
                    indices.append(i)
            new_valid_dataset = torch.utils.data.Subset(valid_dataset, indices)
            valid_dataset = new_valid_dataset

            indices = []
            for i in range(len(test_dataset)): # same process for test
                data, label = test_dataset[i]
                if label % 2 != 0:
                    indices.append(i)
            new_test_dataset = torch.utils.data.Subset(test_dataset, indices)
            test_dataset = new_test_dataset
        # label noise2
        elif args.noise == 3:
            new_train_dataset = []
            train_dataset.targets = torch.tensor(train_dataset.targets)
            for i in range(len(train_dataset)):
                feature, label = train_dataset[i]
                if int(i / 200) % 4 == 0: #This condition is true for every 200th data point 
                    noiselabel = (label + 1) % 10 #adding 1 to the original label and then taking the result modulo 10. This operation ensures the new label is still within the 0-9 range 
                    new_train_dataset.append((feature, noiselabel)) #For index 0, label 0 becomes (0 + 1) % 10 = 1
                else:
                    new_train_dataset.append((feature, label))
            train_dataset = new_train_dataset
        # data noise
        elif args.noise == 4:
            new_train_dataset = []
            for i in range(len(train_dataset)):
                feature, label = train_dataset[i]
                if int(i / 200) % 2 == 0:
                    noise = torch.tensor(np.random.normal(0, 1, feature.shape)) #Gaussian) distribution with a mean of 0 and a standard deviation of 1
                    noise = noise.to(torch.float32)
                    new_data = feature + noise #The noise tensor is added to the original feature tensor
                    clip_data = torch.clamp(new_data, -1, 1) #modified feature values remain within a valid range
                    new_train_dataset.append((clip_data, label))
                else:
                    new_train_dataset.append((feature, label))
            train_dataset = new_train_dataset
            # Add this block after your existing noise conditions
        # elif args.noise == 5:  # Label flipping as a new noise condition
        #     new_train_dataset = []
        #     train_dataset.targets = torch.tensor(train_dataset.targets)
        #     for i, (feature, label) in enumerate(train_dataset):
        # # Define your label flipping logic here
        # # Example: Flip label to the next label in a cyclic manner
        #     flipped_label = (label + 1) % 10  # Assuming 10 classes, labels cycle from 0 to 9
        #     new_train_dataset.append((feature, flipped_label))
    # Update train_dataset with the modified data (features and flipped labels)
    # train_dataset.data = [data[0] for data in new_train_dataset]  # Update features
    # train_dataset.targets = [data[1] for data in new_train_dataset]  # Update labels with flipped ones

    return train_dataset, valid_dataset, test_dataset, user_groups

def add_gradient_noise(args, w, idxs):
    if (args.noise < 5):
        return w
    for i in range(len(w)):
        for key in w[i].keys():
            if idxs[i] % 4 == 0: # noise is added to every fourth parameter set.
                if args.noise == 5:
                    noise = torch.tensor(np.random.normal(0, args.noiselevel, w[i][key].shape)).to(args.device) #noise from guassian distribution with mean of 0 and SD of arg.noiselevel
                    ratio = torch.ones(w[i][key].shape).to(args.device)
                    w[i][key] = w[i][key] * (ratio + noise) #apply noise to weight, weights are multiplying by (1+noise), ratio is a tensor of 1
                if args.noise == 6:
                    noise = torch.tensor(np.random.normal(0, args.noiselevel, w[i][key].shape))
                    noise = noise.to(torch.float32)
                    noise = noise.to(args.device)
                    # print("original weight = ", w[i][key])
                    w[i][key] = noise
                if args.noise == 7:
                    w[i][key] = w[i][key] * -10 #gradient sign inversion
                if args.noise == 8:
                    w[i][key] = torch.ones(w[i][key].shape) * -1 # constant weight corruption 
                if args.noise == 9:  # Scaling Noise
                    w[i][key] = w[i][key] * -100 #gradient sign inversion
                if args.noise == 10: 
                    scale_factor = 1.0 + (torch.randn(w[i][key].size()) * args.noiselevel)
                    w[i][key] *= scale_factor.to(w[i][key].device)

                # if args.noise == 9:
                #     if args.epoch == 10 and int(idxs[i]) in [1, 2, 3]: 
                #         print("Attack_client:", idxs[i])
                #         # Assuming global_weights is accessible here; you may need to pass it as an argument
                #         boosting_factor = 10  # Adjust this formula as needed
                #         mal_weights = boosting_factor * (w[i][key] - global_weights[key])
                #         w[i][key] = mal_weights
    return w
# def add_gradient_noise_to_single_client(args, single_weight, client_idx):
#     # Create a temporary structure that mimics the original expected structure
#     temp_weights = [single_weight]  # Wrap single_weight in a list
#     temp_idxs = [client_idx]  # Wrap client_idx in a list
    
    # Call the original function with the temporary structures
    modified_weights = add_gradient_noise(args, temp_weights, temp_idxs)
    
    # Return the modified weight from the list
    return modified_weights[0]
def add_random_gradient(args, w, idxs):
    for i in range(len(w)):
        for key in w[i].keys():
            if idxs[i] % 10 == 0:
                # print(idxs[i])
                noise = torch.tensor(np.random.normal(0, args.noiselevel, w[i][key].shape))
                noise = noise.to(torch.float32)
                noise = noise.to(args.device)
                # print("original weight = ", w[i][key])
                w[i][key] = noise
                # print("noise weight = ", w[i][key])
    return w

def average_weights(w):
    """
    最正常的平均
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def avgSV_weights(w, shapley,ori):
    """
        Shapley权值平均
        Returns the average of the weights.
    """
    w_avg = copy.deepcopy(ori)
    for key in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[key] = w_avg[key] + (w[i][key]-ori[key]) * shapley[i]
    return w_avg

def avgSV_baseline(w, shapley, ori):
    """
        FedSV Shapley权值平均 beta=0.5
        Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * shapley[0]
        for i in range(1, len(w)):
            w_avg[key] = w_avg[key] + w[i][key] * shapley[i]
        w_avg[key] = torch.div(w_avg[key], 2) + torch.div(ori[key], 2)
    return w_avg


"""
    p: The probabilities of each arm been picked in the first round
    C: The number of arms that been picked in each round
"""
def arms_selection(p,C):
    selected = []
    tuples = []
    for i in range(len(p)):
        tuples.append((i,p[i]))
    remain = 1
    for _ in range(C):
        rand = random.random()
        pre = 0
        for i in range(len(tuples)):
            if tuples[i][0] not in selected:
                if rand >= pre and rand < pre+tuples[i][1]/remain:
                    selected.append(i)
                    remain -= tuples[i][1]
                    break
                else:
                    pre += tuples[i][1]/remain
    return selected

def unbiased_selection(p):
    idxs = []
    while(len(idxs) < 2):
        idxs = []
        for i in range(len(p)):
            rand = random.random()
            if rand < p[i]:
                idxs.append(i)
    return idxs

def softmax(a,eta):
    s = 0
    p = np.zeros(len(a))
    for i in range(len(a)):
        s += math.exp(eta*a[i])
    for i in range(len(a)):
        p[i] = math.exp(eta*a[i])/s
    return p



def exp_details(args):
    print('\nExperimental details:')
    if args.gpu:
        print(f'    Environment   : CUDA {args.gpu}')
    else:
        print(f'    Environment   : CPU')
    print(f'    Dataset   : {args.dataset}')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    NoiseWord = get_noiseword()
    print('    Noise parameters:')
    if args.noise:
        print(f'    Noise  : {NoiseWord[args.noise]}')
        print(f'    NoiseLevel   : {args.noiselevel}')

    return
