import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

class DeepKriging(nn.Module):
    def __init__(self, input_size):
        super(DeepKriging, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(100),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.model(x)

class DeepKrigingTrainer:
    def __init__(self, deposit_data, covariates=None, regular_nn=False, plot_errors=True):
        self.deposit_data = deposit_data
        self.covariates = covariates
        self.phi_columns = deposit_data.columns[10:].tolist()
        if regular_nn:
            self.phi_columns = ['X', 'Y', 'Z']
        if covariates is not None:
            self.p = len(covariates) + len(self.phi_columns)
        else:
            self.p = len(self.phi_columns)
        self.plot_errors = plot_errors
        self.test_mse_list = []
        self.test_mae_list = []
        self.test_adjusted_r2_list = []

    def train_model(self, model, optimizer, criterion, x_train, y_train, x_test, y_test):
        train_losses = []  # To store training losses during training
        test_losses = []   # To store test losses during training

        # Training loop
#        for step in range(601):
        for step in range(601):
            pre = model(x_train)
            mse = criterion(pre, y_train)
            cost = mse

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            pre_test = model(x_test)
            mse_test = criterion(pre_test, y_test)
            test_losses.append(mse_test.item())

            train_losses.append(mse.item())

        if self.plot_errors:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train MSE')
            plt.plot(test_losses, label='Test MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('Training and Test MSE Convergence')
            plt.legend()
            plt.grid(True)
            plt.show()

    def train_neural_network(self):
        torch.manual_seed(42)
        np.random.seed(42)

        # Define the number of folds for cross-validation
        num_folds = 10
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        # Perform k-fold cross-validation
        for fold, (train_index, test_index) in enumerate(kf.split(self.deposit_data)):
            train_data, test_data = self.deposit_data.iloc[train_index], self.deposit_data.iloc[test_index]

            if self.covariates is not None:
                x_train = torch.tensor(train_data[self.phi_columns + self.covariates].values, dtype=torch.float32)
                x_test = torch.tensor(test_data[self.phi_columns + self.covariates].values, dtype=torch.float32)
            else:
                x_train = torch.tensor(train_data[self.phi_columns].values, dtype=torch.float32)
                x_test = torch.tensor(test_data[self.phi_columns].values, dtype=torch.float32)

            y_train = torch.tensor(train_data['Density_gcm3'].values.reshape(-1, 1), dtype=torch.float32)
            y_test = torch.tensor(test_data['Density_gcm3'].values.reshape(-1, 1), dtype=torch.float32)

            model = DeepKriging(self.p)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.005)

            self.train_model(model, optimizer, criterion, x_train, y_train, x_test, y_test)

            # Store metrics for this fold
            test_predictions_fold = model(x_test).detach().numpy().flatten()
            self.test_mse_list.append(mean_squared_error(y_test, test_predictions_fold))
            self.test_mae_list.append(mean_absolute_error(y_test, test_predictions_fold))


            y_test = test_data['Density_gcm3'].values

            # Calculate adjusted R-squared
            n = len(y_test)
            mean_y_test = np.mean(y_test)
            sst = np.mean((y_test - mean_y_test) ** 2)
            ssr = np.mean((test_predictions_fold - y_test) ** 2)
            r2 = 1 - (ssr / sst)
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - self.p - 1))

            self.test_adjusted_r2_list.append(adjusted_r2)
        
            # Print metrics for the current fold
            #print(f"\nMetrics for Fold {fold + 1}:")
            #self.print_metrics(y_test.numpy(), test_predictions_fold, "Test", self.p)

        # Print average metrics across folds
        print("\nAverage Metrics Across Folds:")
        print(f"  Average MSE: {np.mean(self.test_mse_list):.4f}")
        print(f"  Average MAE: {np.mean(self.test_mae_list):.4f}")
        print(f"  Average adjusted R2: {np.mean(self.test_adjusted_r2_list):.4f}")
        print(f"  STD MSE: {np.std(self.test_mse_list):.4f}")
        print(f"  STD MAE: {np.std(self.test_mae_list):.4f}")
        print(f"  STD adjusted R2: {np.std(self.test_adjusted_r2_list):.4f}")

    def print_metrics(self, true_values, predicted_values, dataset_name, p):
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        r2 = r2_score(true_values, predicted_values)
        print(f"{dataset_name} MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")