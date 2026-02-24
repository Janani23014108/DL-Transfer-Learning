# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Include the problem statement and Dataset


## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 
Import required libraries and define image transforms.

### STEP 2: 
Load training and testing datasets using ImageFolder.


### STEP 3: 
Visualize sample images from the dataset.


### STEP 4: 
 Load pre-trained VGG19, modify the final layer for binary classification, and freeze feature extractor layers.

### STEP 5: 
Define loss function (BCEWithLogitsLoss) and optimizer (Adam). Train the model and plot the loss curve.


### STEP 6: 
Evaluate the model with test accuracy, confusion matrix, classification report, and visualize predictions.


## PROGRAM

### Name: J.JANANI

### Register Number: 212223230085

```python
# Load Pretrained Model and Modify for Transfer Learning

from torchvision.models.vgg import VGG19_Weights
from torchvision.models.mobilenetv3 import Weights
model=models.vgg19(weights=VGG19_Weights.DEFAULT)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
summary(model, input_size=(3, 224, 224))

# Modify the final fully connected layer to match the dataset classes

model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,1)


# Include the Loss function and optimizer

criterion =nn.BCEWithLogitsLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)


# Train the model


def train_model(model, train_loader,test_loader,num_epochs=100):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # compute validation loss
        model.eval()
        val_loss = 0.0 
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()
        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: BAVYA SRI B ")
    print("Register Number: 212224230034")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="1072" height="830" alt="image" src="https://github.com/user-attachments/assets/2b4a093a-204e-4f85-8da0-b870bd714359" />


## Confusion Matrix

<img width="886" height="672" alt="image" src="https://github.com/user-attachments/assets/e89ffb6b-53f0-4393-a021-0f1247cce55b" />



## Classification Report
<img width="598" height="252" alt="image" src="https://github.com/user-attachments/assets/df669ed6-ef7a-491d-a83c-9778b332a904" />

### New Sample Data Prediction
<img width="516" height="515" alt="image" src="https://github.com/user-attachments/assets/b62a6ec4-2e62-4b86-91ef-70a2390b26d6" />
<img width="495" height="497" alt="image" src="https://github.com/user-attachments/assets/a4a90adb-1f68-4a7f-b94e-36a4225ff037" />


## RESULT
Developing a Neural Network Classification Model using Transfer Learning was Successfully built
