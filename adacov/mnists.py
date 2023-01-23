import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb

from optim import AdaCov

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.fc1 = nn.Linear(5*5*32, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train(args, model, device, train_loader, optimizer, epoch, test_loader):
    model.train()
    loss = nn.CrossEntropyLoss()
    cum_loss = 0.0
    last_log_idx = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, target)

        l.backward()
        optimizer.step()

        cum_loss += l.item()

        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            cum_loss /= batch_idx - last_log_idx

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), cum_loss))
            
            wandb.log({'train/loss': cum_loss}, step=epoch*len(train_loader) + batch_idx)
            cum_loss = 0.0
            last_log_idx = batch_idx
            test(model, device, test_loader)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    err_rate = 100 * (len(test_loader.dataset) - correct) / len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Error: {}/{} ({:.2f}%)'.format(
        test_loss, len(test_loader.dataset)-correct, len(test_loader.dataset),
        err_rate))
    wandb.log({'val/loss': test_loss, 'val/err': err_rate}, commit=False)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=150, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cuda')

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'pin_memory': True, 'num_workers': 1}
    test_kwargs = {'batch_size': args.test_batch_size, 'pin_memory': True, 'num_workers': 1}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Network().to(device)
    
    #optimizer = optim.Adadelta(model.parameters(), lr=1e-1)
    #optimizer = optim.SGD(model.parameters(), lr=1e-1)    
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = AdaCov(model.parameters(), lr=1e-3, block_size=1024)

    wandb.init(
        project='AdaCov-MNIST',
        name='AdaCov 1e-3 1024 SM (big init)',
    )

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, test_loader)
        scheduler.step()
        print('')


if __name__ == '__main__':
    main()