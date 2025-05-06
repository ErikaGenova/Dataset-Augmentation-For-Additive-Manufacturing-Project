import glob
import os
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from PIL import Image


# Add your custom dataset class here
class DefectsDataset(Dataset):
    def __init__(self, data_dir: str, split: str, transform: Optional[Callable] = None, val_split: float = 0.2, random_seed: int = 42):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = ['NoDefects', 'Defects']
        self.file_paths, self.labels = self._load_data()
        
        # Split train/val
        train_idx, val_idx = train_test_split(
            list(range(len(self.file_paths))),
            test_size=val_split,
            stratify=self.labels,
            random_state=random_seed
        )
        
        if split == 'train':
            selected_idx = train_idx
        else:
            selected_idx = val_idx

        self.file_paths = [self.file_paths[i] for i in selected_idx]
        self.labels = [self.labels[i] for i in selected_idx]
    
    def _load_data(self):
        file_paths = []
        labels = []
        for idx, cls in enumerate(self.classes):
            folder = os.path.join(self.data_dir, cls)
            for ext in ('png', 'jpg', 'jpeg'):
                files = glob.glob(os.path.join(folder, f'*.{ext}'))
                file_paths += files
                labels += [idx] * len(files)
        
        # Print the total number of images loaded
        print(f"Total images loaded: {len(file_paths)}")

        return file_paths, labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, 0.0  # Returning dummy label because VAE needs only the image


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        print(f"Setting up datasets for stage: {stage}")
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5839], std=[0.2074])  # your computed mean and std
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5839], std=[0.2074])
        ])

        self.train_dataset = DefectsDataset(
            data_dir=self.data_dir,
            split='train',
            transform=train_transforms,
        )

        self.val_dataset = DefectsDataset(
            data_dir=self.data_dir,
            split='val',
            transform=val_transforms,
        )

#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        # train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                       transforms.CenterCrop(148),
        #                                       transforms.Resize(self.patch_size),
        #                                       transforms.ToTensor(),])
        
        # val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
        #                                     transforms.CenterCrop(148),
        #                                     transforms.Resize(self.patch_size),
        #                                     transforms.ToTensor(),])
        
        # self.train_dataset = MyCelebA(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )
        
        # # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        # )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     