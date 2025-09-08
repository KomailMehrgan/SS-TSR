import torch
from torch.utils.data import Dataset


class DatasetFromPT(Dataset):
    """
    Custom PyTorch Dataset that loads data from a preprocessed .pt file.
    The .pt file is expected to contain a dictionary with keys:
    'lr_images', 'hr_images', and 'labels'.
    """

    def __init__(self, pt_file_path):
        super(DatasetFromPT, self).__init__()

        print(f"===> Loading preprocessed data from: {pt_file_path}")
        # Load the entire dataset into memory
        self.data = torch.load(pt_file_path)
        self.lr_images = self.data['lr_images']
        self.hr_images = self.data['hr_images']
        self.labels = self.data['labels']
        print(f"===> Data loaded. Found {len(self.labels)} samples.")

    def __getitem__(self, index):
        """
        Returns a single sample from the dataset.
        """
        lr_tensor = self.lr_images[index]
        hr_tensor = self.hr_images[index]
        label = self.labels[index]

        return lr_tensor, hr_tensor, label

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)

