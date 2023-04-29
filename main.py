# %%
import torch
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import numpy as np

def set_hyperparameters():
    return {
        "num_epochs": 1500,
        "momentum": 0,
        "train_frac": 0.5,
        "learning_rate": 3,
        "weight_decay": 0.003,
        "hidden_dim": 256#128
    }

class Model(nn.Module):
    P = 53
    
    def __init__(self, hidden_dim=256):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(self.P, hidden_dim)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, self.P)

    def forward(self, x):
        x = self.embedding(x).flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = self.readout(x)
        return x

def prepare_data(P=53, error_rate=0):
    X = torch.cartesian_prod(torch.arange(P), torch.arange(P))
    y = (X[:, 0] + X[:, 1]) % P
    shuffle = torch.randperm(len(X))
    X, y = X[shuffle], y[shuffle]
    X_train, X_val = X[: int(hyperparameters["train_frac"] * len(X))], X[int(hyperparameters["train_frac"] * len(X)) :]
    y_train, y_val = y[: int(hyperparameters["train_frac"] * len(y))], y[int(hyperparameters["train_frac"] * len(y)) :]

    num_errors = int(error_rate * len(y_train))
    error_indices = torch.randint(0, len(y_train), (num_errors,))
    error_values = torch.randint(0, P, (num_errors,))
    # make sure the new values are different from the old ones
    for i, error_idx in enumerate(error_indices):
        while error_values[i] == y_train[error_idx]:
            error_values[i] = torch.randint(0, P)
    
    y_train[error_indices] = error_values

    return X_train, X_val, y_train, y_val, error_indices


def train_model(model, X_train, y_train, X_val, y_val, error_indices, num_epochs, logger=None):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"], momentum=hyperparameters["momentum"]
    )
    criterion = nn.CrossEntropyLoss()
    results = []

    pbar = tqdm.trange(num_epochs, leave=True, position=0)

    for epoch in pbar:
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_val = model(X_val)
            val_loss = criterion(y_pred_val, y_val)
            val_acc = (y_pred_val.argmax(dim=1) == y_val).float().mean() * 100

        msg = "{:10.2f} | {:>8.2f}, {:>4.0f}".format(loss, val_loss, val_acc)
        pbar.set_description(msg)

    # Compute train accuracies at the end
    with torch.no_grad():
        train_acc = (y_pred.argmax(dim=1) == y_train).float().mean() * 100
        randomized_acc = (y_pred[error_indices].argmax(dim=1) == y_train[error_indices]).float().mean() * 100
        non_randomized_indices = torch.tensor(list(set(range(len(y_train))) - set(error_indices.tolist())))
        non_randomized_acc = (y_pred[non_randomized_indices].argmax(dim=1) == y_train[non_randomized_indices]).float().mean() * 100

    results.append({
        'epoch': epoch,
        'train_loss': loss.item(),
        'train_acc': train_acc.item(),
        'val_loss': val_loss.item(),
        'val_acc': val_acc.item(),
        'randomized_acc': randomized_acc.item(),
        'non_randomized_acc': non_randomized_acc.item()
    })

    return results


if __name__ == "__main__":
    num_runs = 3
    error_rates = np.linspace(0, 0.15, 20)
    test_error_rates_runs = []
    train_error_rates_runs = []
    randomized_error_rates_runs = []
    non_randomized_error_rates_runs = []

    for run in range(num_runs):
        print(f"Run {run + 1}")
        torch.manual_seed(run)
        hyperparameters = set_hyperparameters()

        test_error_rates = []
        train_error_rates = []
        randomized_error_rates = []
        non_randomized_error_rates = []

        for error_rate in error_rates:
            X_train, X_val, y_train, y_val, error_indices = prepare_data(error_rate=error_rate)
            model = Model(hidden_dim=hyperparameters["hidden_dim"])
            results = train_model(model, X_train, y_train, X_val, y_val, error_indices, hyperparameters["num_epochs"], logger=None)
            results_df = pd.DataFrame(results)
            test_accuracy = results_df['val_acc'].iloc[-1]
            train_accuracy = results_df['train_acc'].iloc[-1]
            randomized_accuracy = results_df['randomized_acc'].iloc[-1]
            non_randomized_accuracy = results_df['non_randomized_acc'].iloc[-1]
            test_error_rate = (100 - test_accuracy) / 100
            train_error_rate = (100 - train_accuracy) / 100
            randomized_error_rate = (100 - randomized_accuracy) / 100
            non_randomized_error_rate = (100 - non_randomized_accuracy) / 100
            test_error_rates.append(test_error_rate)
            train_error_rates.append(train_error_rate)
            randomized_error_rates.append(randomized_error_rate)
            non_randomized_error_rates.append(non_randomized_error_rate)

        test_error_rates_runs.append(test_error_rates)
        train_error_rates_runs.append(train_error_rates)
        randomized_error_rates_runs.append(randomized_error_rates)
        non_randomized_error_rates_runs.append(non_randomized_error_rates)

    test_error_rates_avg = np.mean(test_error_rates_runs, axis=0)
    test_error_rates_std = np.std(test_error_rates_runs, axis=0)
    train_error_rates_avg = np.mean(train_error_rates_runs, axis=0)
    train_error_rates_std = np.std(train_error_rates_runs, axis=0)
    randomized_error_rates_avg = np.mean(randomized_error_rates_runs, axis=0)
    randomized_error_rates_std = np.std(randomized_error_rates_runs, axis=0)
    non_randomized_error_rates_avg = np.mean(non_randomized_error_rates_runs, axis=0)
    non_randomized_error_rates_std = np.std(non_randomized_error_rates_runs, axis=0)

    plt.ylim(0,0.15)
    plt.errorbar(error_rates, train_error_rates_avg, yerr=train_error_rates_std, fmt='o', capsize=5, label='Train Error Rate')
    plt.errorbar(error_rates, test_error_rates_avg, yerr=test_error_rates_std, fmt='o', capsize=5, label='Test Error Rate')
    plt.errorbar(error_rates, randomized_error_rates_avg, yerr=randomized_error_rates_std, fmt='o', capsize=5, label='Randomized Train Error Rate')
    plt.errorbar(error_rates, non_randomized_error_rates_avg, yerr=non_randomized_error_rates_std, fmt='o', capsize=5, label='Non-Randomized Train Error Rate')
    plt.plot(error_rates, error_rates, 'k--')
    plt.xlabel('Error Rate')
    plt.ylabel('Error Rate')
    plt.title('Train, Test, Randomized, and Non-Randomized Error Rates vs Error Rate')
    plt.legend()
    plt.show()
# %%
plt.errorbar(error_rates, train_error_rates_avg, yerr=train_error_rates_std, fmt='o', capsize=5, label='Train Error Rate')
plt.errorbar(error_rates, test_error_rates_avg, yerr=test_error_rates_std, fmt='o', capsize=5, label='Test Error Rate')
plt.errorbar(error_rates, randomized_error_rates_avg, yerr=randomized_error_rates_std, fmt='o', capsize=5, label='Randomized Train Error Rate')
plt.errorbar(error_rates, non_randomized_error_rates_avg, yerr=non_randomized_error_rates_std, fmt='o', capsize=5, label='Non-Randomized Train Error Rate')
plt.plot(error_rates, train_error_rates_avg, 'r-', label='Train Error Rate Interpolation')
plt.plot(error_rates, test_error_rates_avg, 'b-', label='Test Error Rate Interpolation')
plt.plot(error_rates, randomized_error_rates_avg, 'g-', label='Randomized Train Error Rate Interpolation')
plt.plot(error_rates, non_randomized_error_rates_avg, 'y-', label='Non-Randomized Train Error Rate Interpolation')
plt.plot(error_rates, error_rates, 'k--')
plt.xlabel('Error Rate')
plt.ylabel('Error Rate')
plt.title('Train, Test, Randomized, and Non-Randomized Error Rates vs Error Rate')
plt.legend()
plt.show()
# %%
