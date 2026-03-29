# Importing required libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from tkinter import *
from tkinter import filedialog
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image

# GUI Setup
main = Tk()
main.title("Land Cover Classification with CNN - PyTorch")
main.geometry("1200x800")
main.config(bg='lightblue')

# Global variables
data_dir = ""
model = None
class_names = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_loader = None
test_loader = None

# GUI Text Output
text = Text(main, height=35, width=90, font=('times', 12))
text.place(x=350, y=50)


def uploadDataset():
    global data_dir, class_names, train_loader, test_loader

    text.delete('1.0', END)

    data_dir = filedialog.askdirectory()
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    class_names = full_dataset.classes

    text.insert(END, f"Dataset loaded from: {data_dir}\n")
    text.insert(END, f"Classes: {class_names}\n")

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_data, test_data = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    text.insert(END, f"\nDataset split into {train_size} training and {test_size} testing images.\n")


def imageProcessing():
    text.insert(END, "\nImage preprocessing is handled using transforms.\n")


def buildTrainModel():
    global model

    text.delete('1.0', END)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 10

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        text.insert(END, f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}\n")
        text.update_idletasks()

    torch.save(model.state_dict(), "cnn_model.pth")
    text.insert(END, "\nModel trained and saved as cnn_model.pth\n")


def evaluateModel():
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds)

    report = classification_report(all_labels, all_preds, target_names=class_names)

    text.insert(END, f"\nAccuracy: {acc:.2f}%\n")
    text.insert(END, f"\nClassification Report:\n{report}\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm,
                annot=True,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues",
                fmt='d')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def predictImage():
    file_path = filedialog.askopenfilename(initialdir=".")

    img = Image.open(file_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)

    label = class_names[pred.item()]

    image_cv = cv2.imread(file_path)
    image_cv = cv2.resize(image_cv, (600, 400))

    cv2.putText(image_cv,
                f"Predicted: {label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Prediction", image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exitApp():
    main.destroy()


def start_gui():
    font = ('times', 13, 'bold')

    Button(main, text="Upload Dataset", command=uploadDataset, font=font).place(x=20, y=50)
    Button(main, text="Image Preprocessing", command=imageProcessing, font=font).place(x=20, y=100)
    Button(main, text="Build & Train CNN", command=buildTrainModel, font=font).place(x=20, y=150)
    Button(main, text="Performance Evaluation", command=evaluateModel, font=font).place(x=20, y=200)
    Button(main, text="Upload Test Image", command=predictImage, font=font).place(x=20, y=250)
    Button(main, text="Exit", command=exitApp, font=font).place(x=20, y=300)


# Let GUI window load first, then build interface
main.after(100, start_gui)
main.mainloop()
