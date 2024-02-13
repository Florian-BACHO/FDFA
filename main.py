from __future__ import print_function
import argparse

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torchvision import datasets, transforms
import torch.optim as optim
import torch.autograd.forward_ad as fwAD

from networks.alexnet_cifar100 import AlexNet
from networks.conv_mnist import Network as NetworkConv_MNIST
from networks.conv_cifar10 import Network as NetworkConv_CIFAR10
from networks.fc_mnist import Network as NetworkFC_MNIST
from networks.fc_cifar10 import Network as NetworkFC_CIFAR10
from utils.MultiOptimizer import MultipleOptimizer
from utils.CSVLogger import CSVLogger

def train(args, model, device, train_loader, optimizer, require_dir_der, epoch, scheduler):
    model.train()

    training_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        if require_dir_der:
            with fwAD.dual_level():
                data = fwAD.make_dual(data, torch.zeros(data.shape, device=device))
                output = model(data)
                output = fwAD.unpack_dual(output).primal
        else:
            output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        optimizer.step()
        # scheduler.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            if args.dry_run:
                break

        training_loss += loss.item()

    training_loss /= (batch_idx + 1)

    return training_loss


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.cross_entropy(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= batch_idx + 1
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.2f}%)\n'.format(
        test_loss, test_accuracy))

    return test_loss, test_accuracy


def main(args_list=None):
    # Training settings
    parser = argparse.ArgumentParser(description='Pytorch Implementation of Backpropagation, '
                                                 'Direct Feedback Alignment, '
                                                 'Direct Kolen-Pollack and Directional DFA.')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--b-lr', type=float, default=1e-4, metavar='BLR',
                        help='learning rate for backward parameters (default: 1e-4)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='Learning rate step gamma (default: 0.95)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='M',
                        help='weight decay (default: 0.0)')
    parser.add_argument('--feedback-decay', type=float, default=0.0, metavar='M',
                        help='feedback decay (default: 0.0, recommended for DKP: 1e-6)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--train-mode', choices=['BP', 'DKP', 'DFA', 'FDFA'], default='FDFA',
                        help='choose between backpropagation (BP), Direct Kolen Pollack (DKP), '
                             'Direct Feedback Alignment (DFA) or Directional DFA (FDFA)')
    parser.add_argument('--log-dir', type=str, default='results/', metavar='DIR',
                        help='directory where metrics will be saved.')
    parser.add_argument('--dataset', choices=['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100'], default='MNIST',
                        help='choose between MNIST, FashionMNIST, CIFAR10 or CIFAR100.')
    parser.add_argument('--conv', action='store_true', default=False,
                        help='train convolutional network.')
    parser.add_argument('--n-layers', type=int, default=2, metavar='N',
                        help='how many hidden layers in the network.')
    parser.add_argument('--seed', type=int, default=-1, metavar='N',
                        help='seed for random generators.')

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        print("Using seed", args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print(args.train_mode)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        kwargs.update({'num_workers': 4,
                       'pin_memory': True})

    transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    if args.dataset == 'MNIST':
        dataset_class = datasets.MNIST
    elif args.dataset == 'FashionMNIST':
        dataset_class = datasets.FashionMNIST
    elif args.dataset == 'CIFAR10':
        dataset_class = datasets.CIFAR10
    else:
        dataset_class = datasets.CIFAR100
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = dataset_class(root='./data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, **kwargs)

    test_data = dataset_class(root='./data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False, **kwargs)

    if args.dataset == "CIFAR100":
        model = AlexNet(args.batch_size, args.train_mode, device).to(device)
    elif args.conv and args.dataset == "CIFAR10":
        model = NetworkConv_CIFAR10(args.batch_size, args.train_mode, device).to(device)
    elif args.conv:
        model = NetworkConv_MNIST(args.batch_size, args.train_mode, device).to(device)
    elif args.dataset == "CIFAR10":
        model = NetworkFC_CIFAR10(args.batch_size, args.n_layers, args.train_mode, device).to(device)
    else:
        model = NetworkFC_MNIST(args.batch_size, args.n_layers, args.train_mode, device).to(device)

    logger = CSVLogger(['Epoch', 'Training Loss', 'Test Loss', 'Test Accuracy'], args)

    if args.train_mode in ['DKP', 'FDFA']:
        # we need to run through the network once to properly initialize the backward weights
        test(model, device, test_loader)

        forward_params = []
        backward_params = []
        for name, param in model.named_parameters():
            print(name, type(param.data), param.size(), param.is_leaf, param.requires_grad)
            if "feedbacks" in name:
                backward_params.append(param)
            else:
                forward_params.append(param)

        forward_optimizer = optim.Adam([{'params': forward_params}], lr=args.lr, weight_decay=args.weight_decay)
        backward_optimizer = optim.Adam([{'params': backward_params}], lr=args.b_lr, weight_decay=args.feedback_decay)
        #backward_optimizer = optim.SGD([{'params': backward_params}], lr=args.b_lr)

        optimizer = MultipleOptimizer(backward_optimizer, forward_optimizer)
        scheduler = StepLR(forward_optimizer, step_size=1, gamma=args.gamma)
    else:
        test(model, device, test_loader)
        for name, param in model.named_parameters():
            print(name, type(param.data), param.size(), param.is_leaf, param.requires_grad)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    require_dir_der = args.train_mode == 'FDFA'

    for epoch in range(1, args.epochs + 1):
        training_loss = train(args, model, device, train_loader, optimizer, require_dir_der, epoch, None)  # scheduler)
        test_loss, test_accuracy = test(model, device, test_loader)

        logger.save_values(epoch, training_loss, test_loss, test_accuracy)
        scheduler.step()


if __name__ == '__main__':
    # Run the FDFA algorithm with a fully-connected DNN with 3 hidden layers on the MNIST dataset
    main(["--n-layers=3", "--dataset=MNIST"])

    # Run the DFA algorithm with a fully-connected DNN with 3 hidden layers on the MNIST dataset
    #main(["--train-mode=DFA", "--n-layers=3", "--dataset=MNIST"])

    # Run the FDFA algorithm with a shallow CNN on the CIFAR10 dataset
    #main(["--conv", "--dataset=CIFAR10"])

    # Run the FDFA algorithm with AlexNet on the CIFAR100 dataset
    #main(["--dataset=CIFAR100", "--batch-size=128"])
