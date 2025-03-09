import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import json
from collections import OrderedDict
import argparse

# Function to load a checkpoint and rebuild the model
def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained = True)
    for param in model.parameters():
        param.requires_grad = False
        
    
    classifier = nn.Sequential(
        nn.Linear(checkpoint['input_size'], checkpoint['hidden_layes'][0]),
        nn.ReLU(),
        nn.Dropout(p=checkpoint['dropout_p']),
        nn.Linear(checkpoint['hidden_layes'][0], checkpoint['output_size']),
        nn.LogSoftmax(dim=1
        ))
     

    
    model.classifier = classifier
    model.load_state_dict(torch.load(filepath), strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    
    
    return model  # Ensure the model is moved to the GPU
def load_image(image_path):
    '''
    load the image from path
    '''
    loaded_image = Image.open(image_path)
    return loaded_image

# Process a PIL image for use in a PyTorch model
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a PyTorch tensor
    '''
    
    # Process a PIL image for use in a PyTorch model
    img = load_image(image)
    
    width, height = img.size
    if width <= height:
        img.thumbnail((256, 256*(height/width)))
    else:
        img.thumbnail((256*(width/height), 256))
   
    width, height = img.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    img = img.crop((left, top, right, bottom))

   

    np_img = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_img = (np_img - mean ) /std
    
    np_img = normalized_img.transpose((2, 0, 1))
    
    return np_img
    
# Predict the class (or classes) of an image using a trained deep learning model
def predict(image_path, model, topk=5):
    
    img = process_image(image_path)
    img = torch.from_numpy(np.array([img])).float()
    
    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    with torch.no_grad():
        model.eval()
        model.cpu()
        output = model.forward(img)
        
    probabilitis = torch.exp(output)
    probs, classes = probabilitis.topk(topk, dim = 1)
    probs  = probs.cpu().numpy().flatten()
    classes = classes.cpu().numpy().flatten()
    
    
    classes = [idx_to_class[i] for i in classes]
    
    return probs, classes
    

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image along with the probability of that name.')
    parser.add_argument('input_image', type=str, help='Path to input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to category names json file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    model.to(torch.device('cuda'))  # Ensure the model is moved to the GPU

    top_probs, top_classes = predict(args.input_image, model, args.top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]

    for prob, cls in zip(top_probs, classes):
        print(f"Class: {cls} - Probability: {prob:.3f}")

if __name__ == '__main__':
    main()


