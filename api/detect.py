import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from io import BytesIO

# Define the create_model function
def create_model(num_classes):
    model = models.resnet50(pretrained=True)
    for param in list(model.parameters())[:-10]:  # Freeze all but last 10 layers
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)  # Replace final layer
    return model

# Load the checkpoint
checkpoint = torch.load(r"C:\Users\Abdullah\Desktop\ResNet\best_model.pth")

# Create the model architecture
model = create_model(num_classes=len(checkpoint['classes']))

# Load the model's state dictionary
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Access the class names
classes = checkpoint['classes']

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Preprocess the image
def preprocess_image(image_data):
    image = Image.open(BytesIO(image_data)).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Make predictions
def predict(image_data, model, classes):
    # Preprocess the image
    image = preprocess_image(image_data)
    
    # Move the image to the same device as the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    model = model.to(device)
    
    # Make a prediction
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        predicted_class = classes[predicted.item()]
    
    return predicted_class

# Function to handle image prediction
def handle_image_prediction(image_data):
    predicted_class = predict(image_data, model, classes)
    print(f'Predicted class: {predicted_class}')  # Print the predicted class to the console
    return predicted_class