from pathlib import Path
import json
import random

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR.parent / "dataset" / "Lung X-Ray Image" / "Lung X-Ray Image"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "lung_model.pt"
LABELS_PATH = MODELS_DIR / "lung_labels.json"
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


class LungXRayDataset(Dataset):
    def __init__(self, root_dir: Path, image_size: int = 128):
        self.root_dir = root_dir
        self.image_size = image_size
        self.class_names = sorted([folder.name for folder in root_dir.iterdir() if folder.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = []
        for class_name in self.class_names:
            class_dir = root_dir / class_name
            for image_path in class_dir.rglob('*'):
                if image_path.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                    self.samples.append((image_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
        tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(self.image_size, self.image_size, 3)
        tensor = tensor.permute(2, 0, 1) / 255.0
        return tensor, label


class SimpleLungCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / total_count, total_correct / total_count


def main():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = LungXRayDataset(DATASET_DIR, image_size=IMAGE_SIZE)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SimpleLungCNN(num_classes=len(dataset.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        sample_count = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            sample_count += labels.size(0)

        train_loss = running_loss / sample_count
        train_acc = running_correct / sample_count
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{EPOCHS} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'class_names': dataset.class_names,
            'image_size': IMAGE_SIZE,
        },
        MODEL_PATH,
    )
    LABELS_PATH.write_text(json.dumps(dataset.class_names, indent=2), encoding='utf-8')
    print(f'Saved lung model to: {MODEL_PATH}')


if __name__ == '__main__':
    main()
