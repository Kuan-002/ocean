import copy
import json
import os
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset
from torchvision import transforms
from tqdm import tqdm
from src.config import Config

def normalize_image(image):
    return (image.float() - 127.5) / 127.5
class CLEVRHansClassification(VisionDataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.image_dir = os.path.join(root, split, "images")
        self.label_file = os.path.join(root, split, f"CLEVR_HANS_scenes_{split}.json")

        # Load labels
        if not os.path.exists(self.label_file):
            raise RuntimeError(f"Label file not found: {self.label_file}")

        with open(self.label_file, "r") as f:
            data = json.load(f)["scenes"]
            self.image_files = [d['image_filename'] for d in data]
            self.targets = [len(d['objects']) for d in data]
            self.classes = {d['image_filename'] : d['class_id'] for d in data}
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.targets[idx]

        if self._transform:
            image = self._transform(image)

        if self._target_transform:
            label = self._target_transform(label)

        return image, label
    
class CLEVRHansN(Dataset):
    def __init__(
        self,
        clevr: CLEVRHansClassification,
        num_obj: int = 6,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        cache: bool = True,
    ):
        super().__init__()
        self.cache = cache
        assert num_obj >= 3 and num_obj <= 10
        self.filter_idx = [i for i, y in enumerate(clevr.targets) if y <= num_obj]
        self._image_files = [f"{clevr.image_dir}/{clevr.image_files[i]}" for i in self.filter_idx]
        self._targets = [clevr.targets[i] for i in self.filter_idx]
        self._transform = transform
        self._target_transform = target_transform
        self._classes = [clevr.classes[clevr.image_files[i]] for i in self.filter_idx]

        if self.cache:
            from concurrent.futures import ThreadPoolExecutor

            self._images = []
            with ThreadPoolExecutor() as executor:
                self._images = list(
                    tqdm(
                        executor.map(self._load_image, self._image_files),
                        total=len(self._image_files),
                        desc=f"Caching CLEVR {clevr.split}",
                        mininterval=(0.1 if os.environ.get("IS_NOHUP") is None else 90),
                    )
                )

    def _load_image(self, file):
        return Image.open(file).convert("RGB")

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx: int):
        if self.cache:
            images = self._images[idx]
        else:
            images = self._load_image(self._image_files[idx])
        targets = self._targets[idx]
        classes = self._classes[idx]

        if self._transform:
            images = self._transform(images)

        if self._target_transform:
            targets = self._target_transform(targets)

        return images, targets, classes

def setup_dataloaders(config: Config, eval: bool = False):
    n = config.dataset_image_resolution * 0.004296875  # 0.55 for 128
    h, w = int(320 * n), int(480 * n)
    aug = {
        "train": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                transforms.RandomCrop(config.dataset_image_resolution),
                transforms.PILToTensor(),
                transforms.Lambda(normalize_image),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                transforms.CenterCrop(config.dataset_image_resolution),
                transforms.PILToTensor(),
                transforms.Lambda(normalize_image),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((h, w), antialias=None),
                transforms.CenterCrop(config.dataset_image_resolution),
                transforms.PILToTensor(),
                transforms.Lambda(normalize_image),
            ]
        ),
    }

    print("Starting, loading dataset.")

    splits = ["train", "val", "test"] if not eval else ["val", "test"]

    datasets = {
        split: CLEVRHansN(
            CLEVRHansClassification(
                root=config.dataset_path,
                split=split
            ),
            num_obj=config.dataset_max_num_obj,
            transform=aug[split],
            cache=config.dataset_cache,
        )
        for split in splits
    }

    dataloaders = {
        split: DataLoader(
            datasets[split],
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            batch_size=config.dataset_batch_size if split == "train" else config.dataset_eval_batch_size,
            num_workers=config.dataset_num_workers,
            pin_memory=True
        )
        for split in splits
    }

    return dataloaders

ch7_classes = {
    0: 'Large (gray) cube and large \ncylinder',
    1: 'Small metal cube and Small \n(metal) sphere',
    2: '(Small) cyan (cube) in \nfront of two red objects',
    3: 'Small green obj. and small \nbrown obj. and small purple obj. \nand two other small obj.s',
    4: '3 spheres on left side or \n3 spheres on left side and 3 metal cyl. on the right side',
    5: 'Three metal cylinders on \nright side',
    6: 'Large blue sphere and small \nyellow sphere',
}

ch3_classes = {
    0: 'Large (gray) cube and \nlarge cylinder',
    1: 'Small metal cube and \nsmall (metal) sphere',
    2: 'Large blue sphere and \nsmall yellow sphere',
}
