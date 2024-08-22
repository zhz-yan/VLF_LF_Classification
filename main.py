import torch
from load_data import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from trainer import *
from net import *

def main(epoch, batch_size):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data (we have done z-normalization according to the article)
    data, labels = load_data()

    # train-test-split
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state=42, test_size=0.2)

    train_data, train_labels = torch.tensor(train_data, dtype=torch.float32).to(device), torch.tensor(train_labels, dtype=torch.long).to(device)
    test_data, test_labels = torch.tensor(test_data, dtype=torch.float32).to(device), torch.tensor(test_labels, dtype=torch.long).to(device)

    # DataLoader
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # define networks
    net = Net().to(device)

    # define optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # train and test
    train(net, train_loader, criterion, optimizer, device, num_epochs=epoch)
    test(net, test_loader)


if __name__ == '__main__':
    epoch = 50
    bz = 32

    main(epoch, bz)
