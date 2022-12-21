import os
from collections import defaultdict

from torch.utils.data import Dataset
from PIL import Image
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
class ImageNetKaggle(Dataset):
    def __init__(self, root, split, restrict_classes = 500, max_samples = 0, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        classes_count = 0
        with open(os.path.join(dir_path, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        if 0 < restrict_classes < classes_count:
                            break
                        self.syn_to_class[v[0]] = int(class_id)
                        classes_count += 1
        print(self.syn_to_class)
        with open(os.path.join(dir_path, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)

        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                if syn_id not in self.syn_to_class:
                    continue
                target = self.syn_to_class[syn_id]

                syn_folder = os.path.join(samples_dir, syn_id)
                samples_count = 0
                for sample in os.listdir(syn_folder):
                    if samples_count > max_samples > 0:
                        break
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
                    samples_count += 1
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                if syn_id not in self.syn_to_class:
                    continue
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]