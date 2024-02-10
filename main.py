#%% Imports
import torchvision.transforms as transforms
from torch import load
from model import BaseModel
from PIL import Image
from visualization import show_mask

#%% Load model
weights_path = "weights/LOCTSeg.ph"
model = BaseModel()
model.load_state_dict(load(weights_path))

#%% Load image
transform = transforms.Compose([transforms.ToTensor()])
img = Image.open("bscan_1.jpg")
img = transform(img)[None,]

# %% model prediction
model.eval()
output = model(img)['out'][0]
show_mask(img[0], output)
