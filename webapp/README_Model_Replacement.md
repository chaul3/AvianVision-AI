# How to Replace Model Weights in the Webapp

## 1. Model Weight Structure Expected

Your trained model should be saved as a PyTorch `.pth` or `.pt` file containing the state_dict of a ResNet50 model with:

- **Input**: 224x224 RGB images (ImageNet normalization)
- **Output**: 312 attribute logits (before sigmoid)
- **Architecture**: ResNet50 backbone + custom classifier head

## 2. Where to Place Your Model

Create a `models/` directory in the webapp folder and place your trained model there:

```
webapp/
├── app.py
├── templates/
├── static/
└── models/              # Create this directory
    ├── bird_attributes_model.pth    # Your trained model weights
    ├── optimal_thresholds.json      # Per-attribute thresholds
    └── species_centroids.npy        # Species attribute centroids
```

## 3. Expected Model Architecture

Your model should match this exact structure:

```python
import torchvision.models as models
import torch.nn as nn

# Base ResNet50
resnet = models.resnet50(pretrained=True)

# Custom classifier head for 312 bird attributes
num_features = resnet.fc.in_features  # 2048
resnet.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(1024, 312)  # 312 CUB bird attributes
)

# Save your trained model
torch.save(resnet.state_dict(), 'bird_attributes_model.pth')
```

## 4. Required Files Format

### A. Model Weights (`bird_attributes_model.pth`)
PyTorch state_dict containing all layer weights

### B. Thresholds (`optimal_thresholds.json`)
```json
{
  "attr_1": 0.42,
  "attr_2": 0.38,
  "attr_3": 0.55,
  ...
  "attr_312": 0.47
}
```

### C. Species Centroids (`species_centroids.npy`)
NumPy array of shape (200, 312) - L2 normalized attribute vectors for each CUB species

## 5. Code Changes Needed in app.py

The main changes will be in the `setup_model()` method to load your trained weights instead of using random initialization.

## 6. Testing Your Model

After replacement, test with:
1. Upload a bird image to the webapp
2. Check the "Show Prompt" section to see attribute predictions
3. Verify similarity scores are no longer 0.000
4. Check that attributes make sense for the bird type

## 7. Cluster Training Output

When training on your cluster, make sure to save:
- Final model weights with torch.save()
- Calibrated thresholds from validation set
- Species centroids computed from training data
- Training logs and metrics for verification
