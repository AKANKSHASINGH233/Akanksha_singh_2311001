import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import SVHN
from sklearn.metrics import precision_score, recall_score, f1_score
import csv

# Step 1: Load the SVHN dataset
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = SVHN(root='./data', split='train', transform=train_transform, download=True)
test_dataset = SVHN(root='./data', split='test', transform=test_transform, download=True)

# Use a subset of the dataset (25%)
train_subset = Subset(train_dataset, range(0, len(train_dataset), 4))

# Step 2: Preprocess the dataset
train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Step 3: Choose pretrained models
# Implement LeNet-5 manually
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

pretrained_models = {
    'LeNet-5': LeNet5(),
    'VGG-16': models.vgg16(pretrained=True),
    'ResNet-18': models.resnet18(pretrained=True),
    'ResNet-50': models.resnet50(pretrained=True),
    'ResNet-101': models.resnet101(pretrained=True)
}

# Step 4: Load the pretrained weights for the chosen model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 5: Fine-tune the model on the SVHN dataset
num_epochs = 10
learning_rate = 0.001

# Initialize CSV writer
with open('model_metrics.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Test Accuracy', 'Precision', 'Recall', 'F1-score'])

    for model_name, model in pretrained_models.items():
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with data augmentation and adjusted hyperparameters
        model.train()  # Set model to training mode
    
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track the loss and accuracy
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            # Print statistics every epoch
            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total * 100
            
            print(f"Model: {model_name}, Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
        print("Training finished for", model_name)
    
        # Evaluate the model on the test set
        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []
    
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
    
        test_accuracy = correct / total * 100
        print(f"Test Accuracy for {model_name}: {test_accuracy:.2f}%")
    
        # Calculate precision, recall, and F1-score
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')
    
        # Write to CSV
        writer.writerow([model_name, test_accuracy, precision, recall, f1])
    
        # Performance report
        print(f"Performance Report for {model_name}:")
        print("--------------------------------------------------")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("--------------------------------------------------")
        print()

print("Output saved to model_metrics.csv")
