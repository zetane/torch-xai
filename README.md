# torch-xai
ZetaForge XAI Torch Pipeline

Block input details:
- test_dataset_file : zip file contaning all the images and xlsx dataframe, where image column name must be "filename".
- model_process_file: pickle file containing model and the pre-process code, see the code below for how to save the model and processor.
- model_architecture_type: three options we have "cnn", "ViT", "SwiT"
- target_layer: name of the target layer on which xai will be performed.
- saving_dir: name of the save dir.

Here is the demo script for how to save model and processor.

```
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import pickle

# Load the ResNet18 model
model = models.resnet50(pretrained=True)
model.eval()

# Create the transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create a dictionary with the model and transform
data = {
    'model': model,
    'processor': transform
}

# Save the dictionary to a pickle file
with open('resnet18_model_and_transform.pkl', 'wb') as f:
    pickle.dump(data, f)
```
