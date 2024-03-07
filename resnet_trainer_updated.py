import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluate_visualization import EvaluateVisualization

# Training class
class ResNetTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, device='cuda'):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.evaluator = EvaluateVisualization()

    def _train_one_epoch(self):
        self.model.train()
        epoch_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)
            _, predicted_train = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        accuracy_train = correct_train / total_train
        return epoch_train_loss / total_train, accuracy_train

    def _validate_one_epoch(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                _, predicted_val = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted_val == labels).sum().item()

        accuracy_val = correct_val / total_val
        return val_loss / total_val, accuracy_val

    def train(self, num_epochs):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            # Training
            train_loss, train_accuracy = self._train_one_epoch()

            # Validation
            val_loss, val_accuracy = self._validate_one_epoch(self.val_dataloader)

            # Print progress
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            # Update the training and validation loss and accuracy lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)

        # Plotting loss curve and accuracy curve
        self.evaluator.plot_loss_curve(train_losses, val_losses)
        self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies)

    def evaluate(self):
        self.model.eval()
        y_true_test = []
        y_pred_test = []
        correct_test = 0
        total_test = 0
        test_loss = 0.0

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                _, predicted_test = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()

                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted_test.cpu().numpy())

                # Calculate test loss
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)

        accuracy_test = correct_test / total_test
        test_loss /= total_test  # Calculate average test loss

        # Plotting confusion matrix
        class_names = self.test_dataloader.dataset.classes
        self.evaluator.plot_confusion_matrix(y_true_test, y_pred_test, class_names)
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy_test:.4f}')
        print('Evaluation finished.')
        
    # def evaluate(self):
    #     self.model.eval()
    #     y_true_test = []
    #     y_pred_test = []
    #     correct_test = 0
    #     total_test = 0

    #     with torch.no_grad():
    #         for inputs, labels in self.test_dataloader:
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             outputs = self.model(inputs)

    #             _, predicted_test = torch.max(outputs.data, 1)
    #             total_test += labels.size(0)
    #             correct_test += (predicted_test == labels).sum().item()

    #             y_true_test.extend(labels.cpu().numpy())
    #             y_pred_test.extend(predicted_test.cpu().numpy())

    #     accuracy_test = correct_test / total_test
    #     # Plotting confusion matrix
    #     class_names = self.test_dataloader.dataset.classes
    #     self.evaluator.plot_confusion_matrix(y_true_test, y_pred_test, class_names)
    #     print(f'Test Accuracy: {accuracy_test:.4f}')
    #     print('Evaluation finished.')
