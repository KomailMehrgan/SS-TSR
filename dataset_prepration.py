import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class DatasetFromImages(Dataset):
    def __init__(self, sr_data_path, scale=1.0):
        super(DatasetFromImages, self).__init__()

        self.scale = scale
        self.sr_data_path = sr_data_path
        self.hr_image_path = os.path.join(sr_data_path, "HR_images")
        self.lr_image_path = os.path.join(sr_data_path, "LR_images")
        self.label_path = os.path.join(sr_data_path, "STR_label")

        # List all HR image files
        self.hr_image_files = [f for f in os.listdir(self.hr_image_path) if os.path.isfile(os.path.join(self.hr_image_path, f))]

        # Create a subset based on the scale
        total_size = len(self.hr_image_files)
        subset_size = int(total_size * self.scale)
        indices = np.random.permutation(total_size)
        subset_indices = indices[:subset_size]

        self.hr_image_files = [self.hr_image_files[i] for i in subset_indices]

        # Extract metadata and corresponding LR image filenames
        self.metadata = [self._extract_metadata(f) for f in self.hr_image_files]
        self.lr_image_files = [self._extract_lr_images(f) for f in self.hr_image_files]

        # Define a transform to convert PIL images to tensors
        self.to_tensor = transforms.ToTensor()

    def _extract_metadata(self, filename):
        # Extract metadata from filename
        basename = os.path.splitext(filename)[0]  # Remove file extension
        parts = basename.split('_')
        N = parts[0]
        N = N.replace("hr","str")+"_"
        label_filename = N + parts[1] + ".txt"
        label_path = os.path.join(self.label_path, label_filename)

        with open(label_path, 'r') as f:
            label = f.read()

        return label

    def _extract_lr_images(self, filename):
        # Extract LR image filename from HR image filename
       # Remove file extension
        filename = filename.replace("hr","lr")

        lr_image_path = os.path.join(self.lr_image_path, filename)
        return lr_image_path

    def __getitem__(self, index):
        # Get the HR and LR image paths
        hr_image_path = os.path.join(self.hr_image_path, self.hr_image_files[index])
        lr_image_path = self.lr_image_files[index]

        # Load the HR and LR images
        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        hr_image= hr_image.resize((128, 32), Image.BICUBIC)
        lr_image= lr_image.resize((64, 16), Image.BICUBIC)

        # Convert images to tensors
        hr_image = self.to_tensor(hr_image)
        lr_image = self.to_tensor(lr_image)

        # Get metadata
        metadata = self.metadata[index]

        return lr_image, hr_image, metadata

    def __len__(self):
        return len(self.hr_image_files)
