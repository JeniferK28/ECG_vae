from models import P_net, Q_net, D_net_gauss
from torch.utils.data import DataLoader
from torch import Tensor
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
from train_pred import pred, train_validate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from utils import read_data_ecg
import argparse
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ECG_reconstruction')
    parser.add_argument("--seed", default=1657, help="Seed number")
    parser.add_argument("--batch_size", default=64, help="Batch_size")
    parser.add_argument("--train_data_path", default='train.hdf5', help="Train data path")
    parser.add_argument("--train_label_path", default='trainlabel.hdf5', help="Train label path")
    parser.add_argument("--test_data_path", default='test.hdf5', help="Train data path")
    parser.add_argument("--test_label_path", default='testlabel.hdf5', help="Train label path")
    parser.add_argument("--lr", default=0.0001, help="learning rate")
    parser.add_argument("--step_size", default=7, help="step size")
    parser.add_argument("--num_epoch", default=100, help="Number of epochs")
    parser.add_argument("--split_size", default=0.2, help="Train and validation split")
    parser.add_argument("--model_path", default='models/', help="Model saving path")
    parser.add_argument("--gen_lr", default=0.0001, help="Generator lr")
    parser.add_argument("--reg_lr", default=0.0001, help="Decoder lr")
    parser.add_argument("--EPS", default=1e-15, help="EPS")
    parser.add_argument("--device", default='cuda', help="device")

    args = parser.parse_args()

    traindata = read_data_ecg(args.train_data_path)
    train_label = read_data_ecg(args.train_label_path)
    train_label = [np.argmax(i) for i in train_label]
    normal_indices = [index for index, value in enumerate(train_label) if value == 0]
    train_data = [np.reshape(traindata[i], (1, 1, 256) ) for i in normal_indices]
    train_label = np.zeros(np.size(normal_indices))

    testdata = read_data_ecg(args.test_data_path)
    test_label = read_data_ecg(args.test_label_path)
    test_label = [np.argmax(i) for i in test_label]
    test_data = [np.reshape(i, (1, 1, 256)) for i in testdata]

    # Split train and val data
    X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=args.split_size, random_state=42)

    train_label= Variable(Tensor(y_train)).type(torch.LongTensor)
    val_label= Variable(Tensor(y_val)).type(torch.LongTensor)
    test_label= Variable(Tensor(test_label)).type(torch.LongTensor)

    train_data = torch.utils.data.TensorDataset(Tensor(X_train), train_label)
    val_data = torch.utils.data.TensorDataset(Tensor(X_val), val_label)
    test_data = torch.utils.data.TensorDataset(Tensor(test_data), test_label)

    train_loader=DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader=DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_loader=DataLoader(test_data, batch_size=1, shuffle=True)


    args.device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define models and send to device
    encoder, decoder =  Q_net(), P_net()     # Encoder/Decoder
    Disc = D_net_gauss()                # Discriminator adversarial

    # Send models to gpu 
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        Disc = Disc.cuda()
        D_gauss = D_net_gauss().cuda()

    loss = []

    # Set optimizators
    optim_decoder = optim.Adam(decoder.parameters(), lr=args.gen_lr)
    optim_encoder = optim.Adam(encoder.parameters(), lr=args.gen_lr)
    optim_D= optim.Adam(Disc.parameters(), lr=args.reg_lr)
    optim_encoder_reg = optim.Adam(encoder.parameters(), lr=args.reg_lr)
    ae_criterion = nn.MSELoss()


    train_loss,train_l1,train_l2 , train_l3, train_labels, val_loss, val_l1, val_l2 , val_l3, val_labels = train_validate(encoder, decoder, Disc, train_loader, val_loader, args, optim_encoder, optim_decoder, optim_D, optim_encoder_reg, ae_criterion, True)
    test_loss,test_l1, test_l2, test_l3, test_labels = pred(test_loader, encoder, decoder, Disc, ae_criterion, args)

    # Outliers
    for i in range(len(val_loss)):
        if val_loss[i]<1:
            loss.append(val_loss[i])

    x=np.mean(loss)
    y=np.std(loss)

    ab_loss=[]
    n_loss=[]

    # Calculating normal and abnormal distribution
    for i in range(len(test_loss)):
             if test_labels[i]>=1:
                 ab_loss.append(test_loss[i])
             else:  n_loss.append(test_loss[i])

    # Lower and upper threshold according to mean and std of validation data
    lower_threshold = np.log(x-y)
    upper_threshold = np.log(x+y)
    print (x,y,lower_threshold, upper_threshold)

    # Save normal and abnormal distribution
    plt.figure(figsize=(12, 6))
    plt.title('Loss Distribution (log)')
    sns.distplot(np.log10(n_loss), kde=True, color='blue')
    sns.distplot(np.log10(ab_loss), kde=True, color='red')
    #plt.axvline(upper_threshold, 0.0, 10, color='r')
    #plt.axvline(lower_threshold, 0.0, 10, color='b')
    plt.savefig('distribution.png')

    tp = 0
    fp = 0
    tn = 0
    fn = 0


    # Calculating confusion matrix
    total_anom = 0
    for i in range(len(test_loss)):
        if test_labels[i]>=1:
            total_anom += 1
        if np.log10(test_loss[i]) >= upper_threshold:
            if float(test_labels[i]) >= 1.0:
                tp += 1
            else:
                fp += 1
        else:
            if float(test_labels[i]) >= 1.0:
                fn += 1
            else:
                tn += 1
    print('[TP] {}\t[FP] {}\t[MISSED] {}'.format(tp, fp, total_anom - tp))
    print('[FN] {}\t[TN] {}'.format(fn, tn))

