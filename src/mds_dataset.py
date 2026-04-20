import os
from PIL import Image
from torchvision.datasets import VisionDataset
from src.config import Config
from torchvision import transforms
from torch.utils.data import DataLoader

class MultiDSprites(VisionDataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        self.image_dir = os.path.join(root, split)
        self.image_files = [img for img in os.listdir(self.image_dir) if img.endswith('.png')]
        self.labels = [int(file_name.split('_')[1]) for file_name in self.image_files]
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, 0, label
    
def setup_dataloaders_mds(config: Config, eval: bool = False):
    aug = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(lambda image: (image - 127.5) / 127.5),
        ]
    )

    print("Starting, loading dataset.")

    splits = ["train", "val", "test"] if not eval else ["val", "test"]

    datasets = { split: MultiDSprites(config.dataset_path, split, aug) for split in splits }

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

mds_classes = {
    0: "A red square and a \nheart.",
    1: "Two hearts on the \nleft side.",
    2: "An ellipse and two \nsquares.",
    3: "A bright object infront \nof a dark background",
    4: "Three different shapes \non the right side."
}
