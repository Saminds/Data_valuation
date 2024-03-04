# Version 2.0
import sys, os
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")

import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from src_opt.utils.options import args_parser
from src_opt.utils.update import LocalUpdate, test_inference
from src_opt.utils.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from src_opt.utils.tools import get_dataset, average_weights, exp_details, add_gradient_noise, add_random_gradient, get_noiseword, add_gradient_noise_new, unbiased_selection
from src_opt.utils.CEXPIX import arms_selection
from src_opt.utils.Shapley import Shapley

args = args_parser()
exp_details(args)

def solver():
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != None else 'cpu')

    # load dataset and user groups
    train_dataset, valid_dataset, test_dataset, user_groups = get_dataset(args) #user_groups map each client to their specific subset of training dataset

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model = global_model.to(args.device)
    global_model.train()

    # copy weights
    global_weights = global_model.state_dict()
    original_weights = copy.copy(global_weights)
    # Training
    train_loss, train_accuracy = [], []
    allAcc_list = []
    print_every = 2
    init_acc = 0

    #The estimated Shapley value of each client
    Phi = np.array([1/args.num_users for _ in range(args.num_users)])#an array containing the estimated Shapley values for each client
    # print("LEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEn: ", len(Phi))
    #The prior probability of each arm been selected in one round
    probabilities = np.array([1/args.num_users for _ in range(args.num_users)]) #prior probabilities of each client being selected for training in a given round.
     
    targeted_clients =[2, 6]
    # targeted_clients =[2, 6, 10, 15]
    #targeted_clients =[2, 6, 10, 15, 50,51]
    mal_local_weights =[]
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1) #Calculates the number of users to participate in the current training round

       #idxs_users = arms_selection(probabilities, m)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model).to(args.device), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        for idx, user_id in enumerate(idxs_users):
            if user_id in targeted_clients:
                # Use idx to index into local_weights since it's aligned with the order of processing
                mal_local_weights = copy.deepcopy(local_weights[idx])
        local_weights_new = add_gradient_noise_new(args, mal_local_weights, user_id)
        local_weights.append(copy.deepcopy(local_weights_new))  

        # local_weights = add_gradient_noise(args, local_weights, idxs_users)
        Fed_sv = Shapley(local_weights, args, global_model, valid_dataset, init_acc)
        shapley = Fed_sv.eval_mcshap(5)
        # print("SHAPLEYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY", len(shapley))
        #update estimated Shapley value
        for i in range(len(shapley)-1):
            #if idxs_users[i] < len(Phi):
            Phi[idxs_users[i]] = Phi[idxs_users[i]]*0.75+shapley[i]*0.25
        #else:
        # Handle the error: idxs_users[i] is out of bounds
            #print(f"Error: idxs_users index {idxs_users[i]} is out of bounds for Phi with length {len(Phi)}")


        #updata probabilities
        sum_exp = 0
        for i in range(len(Phi)):
            sum_exp += np.exp(Phi[i])
        for i in range(len(probabilities)):
            probabilities[i] = np.exp(Phi[i])/sum_exp

        # update global weights
        # local_weights.append(copy.deepcopy(global_weights))

        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        original_weights = copy.copy(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        allAcc_list.append(test_acc)
        print(" \nglobal accuracy:{:.2f}%".format(100 * test_acc))
        init_acc = test_acc

    #draw(args.epochs, allAcc_list, "FedAvg 10 100")
    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    directory = '../save/objects'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = f'{directory}/{args.dataset}_{args.model}_{args.epochs}__Clients[{args.num_users}]_{args.noise}_s_fedavg_scaling.pkl'
    
    with open(file_name, 'wb') as f:
        pickle.dump({'train_loss': train_loss, 'train_accuracy': train_accuracy, 'test_accuracy': allAcc_list}, f)


    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))
    return test_acc, train_accuracy[-1], allAcc_list

def show_avg(list):
    ans = []
    ans.append(np.mean(list[17:22]))
    ans.append(np.mean(list[37:42]))
    ans.append(np.mean(list[57:62]))
    ans.append(np.mean(list[77:82]))
    ans.append(np.mean(list[95:]))
    print(ans)


if __name__ == '__main__':
    test_acc, train_acc = 0, 0
    repeat = 1
    noise = args.noise
    NoiseWord = get_noiseword()
    for _ in range(repeat):
        print("|---- Repetition {} ----|".format(_ + 1))
        test, train, acc_list = solver()
        test_acc += test
        train_acc += train
        show_avg(acc_list)
        path = '../save_opt/{}/FedSV_{}_cnn_E{}_N{}_repeat{}_{}.txt'.format(NoiseWord[noise], args.dataset,
                                                                                    args.epochs,
                                                                                    args.noiselevel, repeat,
                                                                                    'gpu' + str(args.gpu)) #args.device)
        f = open(path, "a+")
        f.writelines("Repetition [%d] : [%s]\n" % (_ + 1, ', '.join(["%.4f" % w for w in acc_list])))
    print('|---------------------------------')
    print("|---- Train Accuracy: {:.2f}%".format(100 * (train_acc / repeat)))
    print("|---- Test Accuracy: {:.2f}%".format(100 * (test_acc / repeat)))
