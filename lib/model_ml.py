""" Model ML related functions """

from typing import Tuple

import pandas as pd
import torch
from torch import nn


class CNNModel(nn.Module):
    """A Conv3D model"""

    def __init__(self, nclasses: int, in_channels: int):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=(3, 3, 3), padding=0),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.fc1 = nn.Linear(483616, 128)
        self.fc2 = nn.Linear(128, nclasses)
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        """Feed forward the data"""
        out = self.conv_layer1(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out

    def save(self, path: str) -> None:
        """Save the model

        Args:
            path : path to save the model
        """
        torch.save(self.state_dict(), path)


def learn(  # pylint: disable=R0914
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    config: dict,
) -> Tuple[nn.Module, pd.DataFrame]:
    """Proceed to the actual learning of a ML model

    Args:
        model : the model
        train_loader : train data
        test_loader : test data
        config : a config file

    Returns:
        The model with learnt parameters and its metrics
    """
    learning_rate = config["learning_rate"]
    n_iters = config["n_iters"]

    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    for _ in range(n_iters):
        for train_images, train_labels in train_loader:
            optimizer.zero_grad()  # clear gradients
            outputs = model(train_images)  # forward prop
            loss = error(outputs, train_labels)  # entropy loss
            loss.backward()  # compute gradients
            optimizer.step()  # update parameters

            count += 1
            if count % 5 == 0:
                # compute accuracy on test sample
                correct = 0
                total = 0
                for test_images, test_labels in test_loader:
                    # Forward propagation
                    outputs = model(test_images)
                    predicted = torch.max(outputs.data, 1)[1]

                    total += len(test_labels)
                    correct += (predicted == test_labels).sum()

                accuracy = 100 * correct / float(total)

                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)

    metrics = pd.DataFrame(
        {"iteration": iteration_list, "loss": loss_list, "accuracy": accuracy_list}
    )

    return model, metrics
