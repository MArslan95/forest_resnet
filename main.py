from custom_dataset import CustomDataset
from evaluate_visualization import EvaluateVisualization
# from resnet_model_101 import ResNet101v2, ResidualBlock
from resnet_50 import ResNet50, ResidualBlock
from resnet_trainer_updated import ResNetTrainer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

def main():
    # Define transformations for data augmentation
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create custom dataset and DataLoader
    data_dir = "./dataset/Training"
    custom_dataset = CustomDataset(data_dir, transform=transform)
    classes = custom_dataset.classes
    print("Classes:", classes)
    train_size = int(0.8 * len(custom_dataset))
    val_size = len(custom_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=12)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=12)

    # Create test dataset and DataLoader
    test_dataset = CustomDataset("./dataset/Test", transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Create ResNet101v2 model
    num_classes = len(custom_dataset.classes)
    # resnet101v2_model = ResNet101v2(num_classes)
    # Create Resnet 50 Model
    resnet50_model = ResNet50(num_classes)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet50_model.parameters(), lr=0.001)

    # Training and evaluation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Fix typo in 'is_available'
    trainer = ResNetTrainer(
        model=resnet50_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )

    num_epochs = 50
    trainer.train(num_epochs)
    trainer.evaluate()

if __name__ == "__main__":
    main()