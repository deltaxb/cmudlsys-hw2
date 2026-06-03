import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
        )),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    blocks = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    blocks += [
        ResidualBlock(hidden_dim, hidden_dim // 2, norm,drop_prob)
        for _ in range(num_blocks)
    ]
    blocks.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*blocks)
    ### END YOUR SOLUTION


def epoch(dataloader, model:nn.Module, opt:ndl.optim.Optimizer=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
    else:
        model.eval()
    aver_loss = 0.0
    aver_err = 0.0
    tot_batch = 0
    for X, y in dataloader:
        logits = model(X)
        loss:ndl.Tensor = nn.SoftmaxLoss()(logits, y)
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
        batch_size = y.shape[0]
        tot_batch += batch_size
        aver_loss += loss.numpy() * batch_size
        aver_err += np.sum(np.argmax(logits.numpy(), axis=-1) != y.numpy())
    aver_loss /= tot_batch
    aver_err /= tot_batch

    return aver_err, aver_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz",
        f"{data_dir}/train-labels-idx1-ubyte.gz"
    )
    tranin_data = ndl.data.DataLoader(train_dataset, batch_size, shuffle=True)
    model = MLPResNet(dim=784, hidden_dim=hidden_dim)
    opt = optimizer(params=model.parameters(),lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_error, train_loss = epoch(tranin_data, model, opt)
    test_dataset = ndl.data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz",
        f"{data_dir}/t10k-labels-idx1-ubyte.gz"
    )
    test_data = ndl.data.DataLoader(test_dataset, batch_size, shuffle=True)
    test_error, test_loss = epoch(test_data, model)
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
