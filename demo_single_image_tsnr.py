import torch
from model.tsrn import TSRN
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms


def transform_(path):
    img = Image.open(path)
    img = img.resize((64, 16), Image.BICUBIC)
    img_tensor = transforms.ToTensor()(img)

    mask = img.convert('L')
    thres = np.array(mask).mean()
    mask = mask.point(lambda x: 0 if x > thres else 255)
    mask = transforms.ToTensor()(mask)
    img_tensor = torch.cat((img_tensor, mask), 0)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


# Initialize the model with the same parameters used during training
model = TSRN(
    scale_factor=2,
    width=128,
    height=32,
    STN=True,  # Since STN layers are present in state_dict
    srb_nums=5,  # 5 SRB blocks as seen in state_dict
    mask=True,  # Likely False since input channels appear to be 3
    hidden_units=32,
)

checkpoint = torch.load("model_best_TPGSR.pth"
                        "",weights_only=False,map_location=torch.device('cpu'))
print("Checkpoint keys:", checkpoint.keys())

# Extract just the model weights (assuming they're in 'state_dict_G')
state_dict = checkpoint['state_dict_G']
model.load_state_dict(state_dict)


# model_object = checkpoint['model']
# state_dict = model_object.state_dict()
# model.load_state_dict(state_dict)

# Set to evaluation mode
model.eval()

# Load and preprocess the image
img_path = "test.png"  # Replace with the path to your input image
images_lr = transform_(img_path)
images_lr = images_lr.to('cpu')

# Perform super-resolution (no text embedding needed)
with torch.no_grad():
    output = model(images_lr)

# Convert the output tensor to a numpy array for displaying
output_image = output.squeeze().permute(1, 2, 0).numpy()
output_image = np.clip(output_image, 0, 1)  # Ensure values are between 0 and 1
output_image = (output_image * 255).astype(np.uint8)  # Convert to 0-255 range for display

# Load the original LR image
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB


# Resize both LR and SR images to 512x64
image_resized = cv2.resize(image, (128, 32))
output_resized = cv2.resize(output_image, (128, 32))
# cv2.imwrite("OUT.png", output_resized)
cv2.imwrite("OUT.png", cv2.cvtColor(output_resized, cv2.COLOR_RGB2BGR))

# Display the original and super-resolved images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Show the original image
axes[0].imshow(image_resized)
axes[0].set_title("Original Image (Resized)")
axes[0].axis('off')

# Show the super-resolved image
axes[1].imshow(output_resized)
axes[1].set_title("Super-Resolved Image (Resized)")
axes[1].axis('off')

plt.show()
