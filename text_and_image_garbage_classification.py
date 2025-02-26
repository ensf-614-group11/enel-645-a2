import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import os
import re
import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer
import wandb

# Define Data Directories
data_dir = "C:/Users/Auste/Documents/ENEL645_GarbageData/"
train_dir = os.path.join(data_dir, "CVPR_2024_dataset_Train")
val_dir = os.path.join(data_dir, "CVPR_2024_dataset_Val")
test_dir = os.path.join(data_dir, "CVPR_2024_dataset_Test")

# Initialize wandb
def initialize_wandb():
    if wandb.run is None:
        wandb.init(
            entity="shcau-university-of-calgary-in-alberta",
            project="transfer_learning_garbage",
            name="Multimodal_Model_RTX4060_R3",
            tags=["distilBERT", "efficientnet", "CVPR_2024_dataset"],
            notes="Multimodal classification model using distilBERT and efficientnet.",
            config={"epochs": 5, "batch_size": 128, "dataset": "CVPR_2024_dataset"},
            job_type="train",
            resume="allow",
        )

initialize_wandb()

# Define transformations
transform = {
    "train": transforms.Compose([
        models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms(), # This includes the following preprocessing: The images are resized to resize_size=[384] using interpolation=InterpolationMode.BILINEAR,
        # followed by a central crop of crop_size=[384]. Finally, the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        transforms.RandomHorizontalFlip(), # additional data augmentation step added to training data set
    ]),
    "val": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms(),
    "test": models.EfficientNet_V2_S_Weights.IMAGENET1K_V1.transforms(),
}

# Load datasets
image_datasets = {
    "train": datasets.ImageFolder(train_dir, transform=transform["train"]),
    "val": datasets.ImageFolder(val_dir, transform=transform["val"]),
    "test": datasets.ImageFolder(test_dir, transform=transform["test"]),
}


# Text Classification

# Extract text from file names as well as labels
def read_text_files_with_labels(path):
    texts = []
    labels = []
    class_folders = sorted(os.listdir(path))
    label_map = {class_name: idx for idx, class_name in enumerate(class_folders)}

    for class_name in class_folders:
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            file_names = os.listdir(class_path)
            for file_name in file_names:
                file_path = os.path.join(class_path, file_name)
                if os.path.isfile(file_path):
                    file_name_no_ext, _ = os.path.splitext(file_name)
                    text = file_name_no_ext.replace('_', ' ')
                    text_without_digits = re.sub(r'\d+', '', text)
                    texts.append(text_without_digits)
                    labels.append(label_map[class_name])

    return np.array(texts), np.array(labels)

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Prepare text data

text_train,labels_train = read_text_files_with_labels(train_dir)
text_val,labels_val = read_text_files_with_labels(val_dir)
text_test,labels_test = read_text_files_with_labels(test_dir)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 24

#Define number of epochs
EPOCHS = 5


class MultimodalDataset(Dataset):
    def __init__(self, image_dataset, text_dataset):
        self.image_dataset = image_dataset
        self.text_dataset = text_dataset

    def __len__(self):
        return min(len(self.image_dataset), len(self.text_dataset))

    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        text_data = self.text_dataset[idx]
        return {
            "image": image,
            "input_ids": text_data["input_ids"],
            "label": label
        }
    

class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes):
        super(MultimodalClassifier, self).__init__()

        # EfficientNet (Image)
        self.image_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

        # Freeze feature layers
        for param in self.image_model.features.parameters():
            param.requires_grad = False

        num_ftrs = self.image_model.classifier[1].in_features

        #Remove EfficientNet classifier
        self.image_model.classifier = nn.Identity()

        #Project features to 256 nodes
        self.image_fc = nn.Linear(num_ftrs, 256)

        # DistilBERT (Text)
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_fc = nn.Linear(self.text_model.config.hidden_size, 256)

        # Normalization layers
        self.text_norms = nn.LayerNorm(256)
        self.image_norm = nn.LayerNorm(256)

        # Feature fusion Layer (Concatenation)
        self.fusion_fc = nn.Linear(512, self.text_model.config.hidden_size)

        # Classification Layer
        self.classifier = nn.Linear(self.text_model.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_ids, image_inputs):
        # Extract features
        text_output = self.text_model(input_ids=input_ids)
        text_features = self.text_norms(self.text_fc(text_output.last_hidden_state[:, 0, :]))  # Use CLS token
        image_features = self.image_norm(self.image_fc(self.image_model(image_inputs)))

        # Concatenate text and image features
        combined_features = torch.cat((text_features, image_features), dim=1)

        combined_features = self.fusion_fc(combined_features)
        output = self.classifier(self.dropout(combined_features))

        return output
    
# Data Loaders
BATCH_SIZE = 128
train_loader = DataLoader(MultimodalDataset(image_datasets["train"], CustomTextDataset(text_train, labels_train, tokenizer, max_len)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(MultimodalDataset(image_datasets["val"], CustomTextDataset(text_val, labels_val, tokenizer, max_len)), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(MultimodalDataset(image_datasets["test"], CustomTextDataset(text_test, labels_test, tokenizer, max_len)), batch_size=BATCH_SIZE, shuffle=False)

# Evaluation Function
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy
        
# Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalClassifier(num_classes=4).to(device)
optimizer = optim.Adam([
    {'params': model.text_model.parameters(), 'lr': 0.0001},  
    {'params': model.image_fc.parameters(), 'lr': 0.001},  
    {'params': model.classifier.parameters(), 'lr': 0.001}
])
criterion = nn.CrossEntropyLoss()

wandb.watch(model, log="all")
best_val_loss = float("inf")

# Training
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    val_loss, val_acc = evaluate_model(model, val_loader, device)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_multimodal_model.pth")

    wandb.log({"epoch": epoch+1, "train_loss": total_train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
    print(f"Epoch {epoch+1}/{5}, Train Loss: {total_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Load Best Model for Testing
model.load_state_dict(torch.load("best_multimodal_model.pth"))
test_loss, test_acc = evaluate_model(model, test_loader, device)
wandb.log({"test_accuracy": test_acc})
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
wandb.finish()