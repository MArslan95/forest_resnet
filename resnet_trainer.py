import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from evaluate_visualization import EvaluateVisualization

# Training class
class ResNetTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer,scheduler, device='cuda'):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.evaluator = EvaluateVisualization()

    def _train_one_epoch(self):
        self.model.train()
        epoch_train_loss = 0.0
        train_samples = 0

        for inputs, labels in self.train_dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_train_loss += loss.item() * inputs.size(0)
            train_samples += inputs.size(0)

        return epoch_train_loss / train_samples

    def _validate_one_epoch(self):
        self.model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for inputs, labels in self.val_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_samples += inputs.size(0)

        return val_loss / val_samples

    def train(self, num_epochs):
        train_losses = []
        val_losses = []
        test_losses = []  # New: To store test losses
        train_accuracies = []
        val_accuracies = []
        test_accuracies = []  # New: To store test accuracies

        for epoch in range(num_epochs):
            # Training
            train_loss,train_accuracy = self._train_one_epoch()

            # Validation
            val_loss,val_accuracy = self._validate_one_epoch()
            # Test 
            test_loss, test_accuracy = self.evaluate()  

            # Print progress
            print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

            # print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Update the training and validation loss lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            test_accuracies.append(test_accuracy)

        # Plotting loss curve
        self.evaluator.plot_loss_curve(train_losses, val_losses)
        self.evaluator.plot_accuracy_curve(train_accuracies, val_accuracies )  # Include test_accuracies
        self.evaluator.plot_test_acc_loss_curve(test_accuracies,test_losses)

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
                _, predicted_test = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()
                y_true_test.extend(labels.cpu().numpy())
                y_pred_test.extend(predicted_test.cpu().numpy())
                
                # Calculate test loss
                loss = self.criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
        accuracy_test = correct_test / total_test
        test_loss /= total_test

        # Plotting confusion matrix
        # class_names = self.test_dataloader.dataset.classes
        # self.evaluator.plot_confusion_matrix(y_true_test, y_pred_test, class_names)
        print('Evaluation finished.')