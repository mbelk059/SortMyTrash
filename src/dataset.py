import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_CLASSES = ["plastic", "metal", "paper", "cardboard", "glass", "organic", "trash"]

class WasteDataset(Dataset):
    def __init__(self, root_dir, split="train", classes=None, transform=None):
        self.root_dir = os.path.join(root_dir, split) if split else root_dir
        if classes is None:
            classes = DEFAULT_CLASSES
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        self.transform = transform

        for c in self.classes:
            class_dir = os.path.join(self.root_dir, c)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
