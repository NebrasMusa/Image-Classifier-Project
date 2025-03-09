import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
import argparse

# Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load the datasets with ImageFolder
def load_datasets(data_dir):
    image_datasets = {
        'train': datasets.ImageFolder(data_dir + '/train', transform=data_transforms['train']),
        'valid': datasets.ImageFolder(data_dir + '/valid', transform=data_transforms['valid']),
        'test': datasets.ImageFolder(data_dir + '/test', transform=data_transforms['test'])
    }
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32)
    }
    return image_datasets, dataloaders

# Build and train your network
def build_model(arch='vgg16', hidden_units=4096):
    
    model = getattr(models, model_name)(pretrained=True) 
    for param in model.parameters():
        param.requires_grad = False
        
    classifier = nn.Sequential (OrderedDict([('input', nn.Linear(25088 , 120)),
                                             ('relu1', nn.ReLU()),
                                             ('dropout1',nn.Dropout(p=0.5)),
                                             ('lay1', nn.Linear(120,90)),
                                             ('relu2', nn.ReLU()),
                                             ('lay2', nn.Linear(90,70)),
                                             ('relu3', nn.ReLU()),
                                             ('lay3', nn.Linear(70,102)),
                                             ('output', nn.LogSoftmax(dim=1))
                                            ]))

    model.classifier = classifier
    if torch.cuda.is_available():
        model.cuda()
    return model

def train_model(model, dataloaders, criterion, optimizer, epochs=5, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    steps = 0
    print_every = 10

    for epoch in range(epochs):
        running_loss = 0
        for images, labels in dataloaders['train']:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                accuracy = 0

                with torch.no_grad():
                    for images, labels in dataloaders['valid']:
                        images, labels = images.to(device), labels.to(device)
                        output = model.forward(images)
                        valid_loss += criterion(output, labels).item()
                        ps = torch.exp(output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                running_loss = 0
                model.train()

def validate_model(model, dataloaders, criterion, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    test_loss = 0
    accuracy = 0

    with torch.no_grad():
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(dataloaders['test']):.3f}.. "
          f"Test accuracy: {accuracy/len(dataloaders['test']):.3f}")

def save_checkpoint(model, image_datasets, save_dir):
    model.class_to_idx = image_datasets['train_datasets'].class_to_idx
    checkpoint = {
        'model' : 'vgg16',
        'input_size' : 25088,
        'output_size' : 105,
        'hidden_layers' : [4096],
        'dropout_p': 0.2,
        'state_dict' : model.state_dict(),
        'class_to_idx' : model.class_to_idx,
        'optimizer' : optimizer.state_dict(),
        'epochs' : epochs
        }

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    torch.save(checkpoint, save_dir + '/checkpoint.pth')

def main():
    parser = argparse.ArgumentParser(description='Train a new network on a dataset and save the model as a checkpoint.')
    parser.add_argument('data_directory', type=str, help='Dataset directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    image_datasets, dataloaders = load_datasets(args.data_directory)
    model = build_model(args.arch, args.hidden_units)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    train_model(model, dataloaders, criterion, optimizer, args.epochs, args.gpu)
    validate_model(model, dataloaders, criterion, args.gpu)
    save_checkpoint(model, image_datasets, args.save_dir)

if __name__ == '__main__':
    main()


