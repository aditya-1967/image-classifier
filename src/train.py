import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image as PILImage

# ImageNet mean/std
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_trasform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

validation_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


train_dataset = datasets.ImageFolder("../data/seg_train/seg_train", transform = train_trasform)
validation_dataset = datasets.ImageFolder("../data/seg_test/seg_test", transform = validation_transform)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 0)
validation_dataloader = DataLoader(validation_dataset, batch_size = 32, shuffle = False, num_workers = 0)

CLASSES = train_dataset.classes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print(f"Classes: {CLASSES}")
print(f"Train: {len(train_dataset)} | Validation; {len(validation_dataset)}")
print(f"Device: {DEVICE}")

# simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes = 6):
        super().__init__()
        self.features = nn.Sequential(
            # block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),

            #block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),

            # block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace = True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))
    

cnn = SimpleCNN(num_classes = 6).to(DEVICE)
params = sum(p.numel() for p in cnn.parameters() if p.requires_grad)
print(f"SimpleCNN Paramterers: {params:,}")

# training
def train_epoch(model, loader, opt, criterion):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def validation_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs)
        total_loss += criterion(logits, labels).item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def fit(model, train_dl, validation_dl, epochs = 15, learning_rate = 1e-3, tag = "model"):
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = learning_rate
    )
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = epochs)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        training_loss, training_accuancy = train_epoch(model, train_dl, opt, criterion)
        validation_loss, validation_accuracy = validation_epoch(model, validation_dl, criterion)

        schedular.step()

        for k, v in zip(history, [training_loss, validation_loss, training_accuancy, validation_accuracy]):
            history[k].append(v)
        if validation_accuracy > best_acc:
            best_acc = validation_accuracy
            torch.save(model.state_dict(), f"best_{tag}.pt")
        print(f"Ep {epoch:02d} | train {training_accuancy:.3f} | val {validation_accuracy:.3f} | best {best_acc:.3f}")

    return history


# transfer learning, ResNet18
def build_resnet(num_classes = 6):
    model = models.resnet18(weights = "IMAGENET1K_V1")
    # freeze entire backbone
    for parameter in model.parameters():
        parameter.requires_grad = False
    # replace final FC layer, only this trains at first
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model


# visualization of CNN
def grad_cam(model, image_tensor, target_layer):
    model.eval()
    activations, gradiatents = [], []

    def foward_hook(_, __, out):
        activations.append(out.detach())
    def backward_hook(_, __, grad_out):
        gradiatents.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(foward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    out = model(image_tensor.unsqueeze(0).to(DEVICE))
    prediction = out.argmax(1)
    model.zero_grad()
    out[0, prediction].backward()

    h1.remove()
    h2.remove()

    acts = activations[0].squeeze(0)
    grads = gradiatents[0].squeeze(0)
    weights = grads.mean(dim = (1, 2))
    cam = (weights[:, None, None] * acts).sum(0).relu().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, CLASSES[prediction.item()]


def show_grad_cam(model, dataset, num_images = 4):
    fig, axes = plt.subplots(2, num_images, figsize=(16, 8))
    for i, (img_t, label) in enumerate(list(dataset)[:num_images]):
        cam, pred = grad_cam(model, img_t, model.layer4[1].conv2)

        # Original image — denormalise
        img_display = img_t.permute(1, 2, 0).numpy()
        img_display = img_display * np.array(STD) + np.array(MEAN)
        img_display = np.clip(img_display, 0, 1)

        # Resize CAM to image size and overlay
        cam_resized = np.array(PILImage.fromarray(np.uint8(cam * 255)).resize(
            (img_display.shape[1], img_display.shape[0]), PILImage.BILINEAR)) / 255.0
        heatmap = cm.jet(cam_resized)[:, :, :3]
        overlay = 0.5 * img_display + 0.5 * heatmap

        axes[0, i].imshow(img_display)
        axes[0, i].set_title(f"True: {CLASSES[label]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f"Pred: {pred}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam.png", dpi=150, bbox_inches="tight")
    print("Saved gradcam.png")
    plt.show()


def plot_history(history_cnn, hist_s1, hist_s2):
    resnet_acc = hist_s1["val_acc"] + hist_s2["val_acc"]
    resnet_loss = hist_s1["val_loss"] + hist_s2["val_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history_cnn["val_acc"], label="SimpleCNN")
    axes[0].plot(resnet_acc, label="ResNet18")
    axes[0].set_title("Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(history_cnn["val_loss"], label="SimpleCNN")
    axes[1].plot(resnet_loss, label="ResNet18")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    print("Saved training_curves.png")
    plt.show()




if __name__ == '__main__':
    history_cnn = fit(cnn, train_dataloader, validation_dataloader, epochs = 15, tag = "cnn_scratch")

    # build ResNet
    resnet = build_resnet().to(DEVICE)

    hist_s1 = fit(resnet, train_dataloader, validation_dataloader, epochs = 5, learning_rate = 1e-3, tag = "resnet_head")

    for p in resnet.parameters():
        p.requires_grad = True
    
    hist_s2 = fit(resnet, train_dataloader, validation_dataloader, epochs = 10, learning_rate = 1e-4, tag = "resnet_full")


    resnet.load_state_dict(torch.load("best_resnet_full.pt"))
    show_grad_cam(resnet, validation_dataset)
    plot_history(history_cnn, hist_s1, hist_s2)
