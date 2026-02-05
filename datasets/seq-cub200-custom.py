# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini,
# Angelo Porrello, Simone Calderara.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom CUB-200-2011 dataset for Mammoth (Colab-friendly).

What this fixes:
- The original seq_cub200 implementation may rely on OneDrive/SharePoint links
  (onedrivedownloader) that often break in Colab due to redirects/HTML responses.
- This version downloads the official CUB_200_2011.tgz from Caltech and builds
  processed NPZ files locally (train_data.npz / test_data.npz) if missing.

Dataset name to use:
  --dataset seq-cub200-custom
"""

import logging
import os
import tarfile
import urllib.request
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from datasets.transforms.denormalization import DeNormalize
from datasets.utils import set_default_from_args
from datasets.utils.continual_dataset import (
    ContinualDataset,
    fix_class_names_order,
    store_masked_loaders,
)
from utils import smart_joint
from utils.conf import base_path


# -----------------------------
# Utilities: download + preprocess
# -----------------------------

_CUB_TGZ_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"


def _ensure_cub_downloaded(root: str) -> str:
    """
    Ensures the official CUB_200_2011 folder exists under `root`.
    Returns the path to the extracted folder: <root>/CUB_200_2011
    """
    cub_dir = smart_joint(root, "CUB_200_2011")
    if os.path.isdir(cub_dir) and len(os.listdir(cub_dir)) > 0:
        logging.info("CUB_200_2011 already present on disk.")
        return cub_dir

    os.makedirs(root, exist_ok=True)
    tgz_path = smart_joint(root, "CUB_200_2011.tgz")

    logging.info("Downloading CUB-200-2011 (official tgz) ...")
    urllib.request.urlretrieve(_CUB_TGZ_URL, tgz_path)

    logging.info("Extracting CUB_200_2011.tgz ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=root)

    try:
        os.remove(tgz_path)
    except OSError:
        pass

    if not os.path.isdir(cub_dir):
        raise RuntimeError(f"Extraction failed: expected folder not found: {cub_dir}")

    logging.info("CUB-200-2011 downloaded and extracted successfully.")
    return cub_dir


def _read_mapping_files(cub_dir: str):
    """
    Reads official CUB files:
    - images.txt
    - image_class_labels.txt
    - train_test_split.txt
    Returns list of (img_rel_path, class_id_0based, is_train).
    """
    images_txt = smart_joint(cub_dir, "images.txt")
    labels_txt = smart_joint(cub_dir, "image_class_labels.txt")
    split_txt = smart_joint(cub_dir, "train_test_split.txt")

    # image_id -> rel_path
    id2path = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        for line in f:
            img_id, rel = line.strip().split(maxsplit=1)
            id2path[int(img_id)] = rel

    # image_id -> class_id (1..200)
    id2cls = {}
    with open(labels_txt, "r", encoding="utf-8") as f:
        for line in f:
            img_id, cls_id = line.strip().split()
            id2cls[int(img_id)] = int(cls_id) - 1  # 0-based

    # image_id -> is_train (1 train / 0 test)
    id2train = {}
    with open(split_txt, "r", encoding="utf-8") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            id2train[int(img_id)] = (int(is_train) == 1)

    items = []
    for img_id in sorted(id2path.keys()):
        items.append((id2path[img_id], id2cls[img_id], id2train[img_id]))
    return items


def _preprocess_to_npz(root: str, img_size: int = 224) -> None:
    """
    Builds train_data.npz and test_data.npz under `root` if they do not exist.

    We store:
    - data: uint8 array [N, H, W, 3] with fixed size (img_size x img_size)
    - targets: int64 array [N]
    - classes: list of class names (strings)
    - segs: dummy zero masks (kept for compatibility)
    """
    train_npz = smart_joint(root, "train_data.npz")
    test_npz = smart_joint(root, "test_data.npz")

    if os.path.isfile(train_npz) and os.path.isfile(test_npz):
        logging.info("Processed NPZ files already exist. Skipping preprocessing.")
        return

    cub_dir = _ensure_cub_downloaded(root)
    items = _read_mapping_files(cub_dir)

    # Deterministic preprocessing: Resize(256) + CenterCrop(img_size)
    preprocess = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
    ])

    def load_and_process(rel_path: str) -> np.ndarray:
        img_path = smart_joint(cub_dir, "images", rel_path)
        img = Image.open(img_path).convert("RGB")
        img = preprocess(img)
        arr = np.array(img, dtype=np.uint8)  # [H,W,3]
        return arr

    train_data, train_targets = [], []
    test_data, test_targets = [], []

    logging.info("Building processed NPZ files (this runs once) ...")
    for rel_path, cls_id, is_train in items:
        arr = load_and_process(rel_path)
        if is_train:
            train_data.append(arr)
            train_targets.append(cls_id)
        else:
            test_data.append(arr)
            test_targets.append(cls_id)

    train_data = np.stack(train_data, axis=0)
    test_data = np.stack(test_data, axis=0)
    train_targets = np.asarray(train_targets, dtype=np.int64)
    test_targets = np.asarray(test_targets, dtype=np.int64)

    # Dummy seg masks (compatibility)
    train_segs = np.zeros((train_data.shape[0], img_size, img_size), dtype=np.uint8)
    test_segs = np.zeros((test_data.shape[0], img_size, img_size), dtype=np.uint8)

    # Classes: use provided CLASS_NAMES list (200 entries)
    classes = np.array(CLASS_NAMES, dtype=object)

    np.savez_compressed(train_npz, data=train_data, targets=train_targets, classes=classes, segs=train_segs)
    np.savez_compressed(test_npz, data=test_data, targets=test_targets, classes=classes, segs=test_segs)

    logging.info(f"Saved: {train_npz} ({train_data.shape[0]} samples)")
    logging.info(f"Saved: {test_npz} ({test_data.shape[0]} samples)")


# -----------------------------
# Dataset classes
# -----------------------------

class MyCUB200(Dataset):
    """
    Colab-friendly CUB200 dataset that ensures local availability and uses NPZ files.
    """
    IMG_SIZE = 224
    N_CLASSES = 200

    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=True) -> None:
        self.not_aug_transform = transforms.Compose([
            transforms.Resize(MyCUB200.IMG_SIZE, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if download:
            # Ensure raw + processed exist
            os.makedirs(root, exist_ok=True)
            _preprocess_to_npz(root, img_size=MyCUB200.IMG_SIZE)

        data_file = np.load(
            smart_joint(root, "train_data.npz" if train else "test_data.npz"),
            allow_pickle=True
        )

        self.data = data_file["data"]  # uint8 [N,H,W,3]
        self.targets = torch.from_numpy(data_file["targets"]).long()
        self.classes = data_file["classes"]
        self.segs = data_file["segs"]
        self._return_segmask = False

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img, mode="RGB")
        not_aug_img = self.not_aug_transform(img.copy())

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, not_aug_img, self.logits[index]] if hasattr(self, "logits") else [
            img, target, not_aug_img
        ]

        if self._return_segmask:
            raise RuntimeError("Unsupported segmentation output in training set!")

        return ret_tuple

    def __len__(self) -> int:
        return len(self.data)


class CUB200(MyCUB200):
    """Base CUB200 dataset (no not-aug image)."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False) -> None:
        super().__init__(root, train=train, transform=transform,
                         target_transform=target_transform, download=download)

    def __getitem__(self, index: int, ret_segmask=False) -> Tuple[Image.Image, int, Image.Image]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        ret_tuple = [img, target, self.logits[index]] if hasattr(self, "logits") else [img, target]

        if ret_segmask or self._return_segmask:
            seg = self.segs[index]
            seg = Image.fromarray(seg, mode="L")
            seg = transforms.ToTensor()(transforms.CenterCrop((MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE))(seg))[0]
            ret_tuple.append((seg > 0).int())

        return ret_tuple


class SequentialCUB200Custom(ContinualDataset):
    """
    Sequential CUB200 Dataset (custom downloader/preprocessor).
    """
    NAME = "seq-cub200-custom"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 20
    N_TASKS = 10
    SIZE = (MyCUB200.IMG_SIZE, MyCUB200.IMG_SIZE)

    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    TRANSFORM = transforms.Compose([
        transforms.Resize((300, 300), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    TEST_TRANSFORM = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(MyCUB200.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    def get_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        train_dataset = MyCUB200(base_path() + "CUB200", train=True,
                                 download=True, transform=SequentialCUB200Custom.TRANSFORM)
        test_dataset = CUB200(base_path() + "CUB200", train=False,
                              download=True, transform=SequentialCUB200Custom.TEST_TRANSFORM)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        return transforms.Compose([transforms.ToPILImage(), SequentialCUB200Custom.TRANSFORM])

    @set_default_from_args("backbone")
    def get_backbone():
        return "vit"

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_normalization_transform():
        return transforms.Normalize(SequentialCUB200Custom.MEAN, SequentialCUB200Custom.STD)

    @staticmethod
    def get_denormalization_transform():
        return DeNormalize(SequentialCUB200Custom.MEAN, SequentialCUB200Custom.STD)

    @set_default_from_args("batch_size")
    def get_batch_size(self):
        return 128

    @set_default_from_args("n_epochs")
    def get_epochs(self):
        return 50

    def get_class_names(self):
        if self.class_names is not None:
            return self.class_names
        classes = fix_class_names_order(CLASS_NAMES, self.args)
        self.class_names = classes
        return self.class_names


# 200 class names (keep as in original file)
CLASS_NAMES = [
    'black footed albatross',
    'laysan albatross',
    'sooty albatross',
    'groove billed ani',
    'crested auklet',
    'least auklet',
    'parakeet auklet',
    'rhinoceros auklet',
    'brewer blackbird',
    'red winged blackbird',
    'rusty blackbird',
    'yellow headed blackbird',
    'bobolink',
    'indigo bunting',
    'lazuli bunting',
    'painted bunting',
    'cardinal',
    'spotted catbird',
    'gray catbird',
    'yellow breasted chat',
    'eastern towhee',
    'chuck will widow',
    'brandt cormorant',
    'red faced cormorant',
    'pelagic cormorant',
    'bronzed cowbird',
    'shiny cowbird',
    'brown creeper',
    'american crow',
    'fish crow',
    'black billed cuckoo',
    'mangrove cuckoo',
    'yellow billed cuckoo',
    'gray crowned rosy finch',
    'purple finch',
    'northern flicker',
    'acadian flycatcher',
    'great crested flycatcher',
    'least flycatcher',
    'olive sided flycatcher',
    'scissor tailed flycatcher',
    'vermilion flycatcher',
    'yellow bellied flycatcher',
    'frigatebird',
    'northern fulmar',
    'gadwall',
    'american goldfinch',
    'european goldfinch',
    'boat tailed grackle',
    'eared grebe',
    'horned grebe',
    'pied billed grebe',
    'western grebe',
    'blue grosbeak',
    'evening grosbeak',
    'pine grosbeak',
    'rose breasted grosbeak',
    'pigeon guillemot',
    'california gull',
    'glaucous winged gull',
    'heermann gull',
    'herring gull',
    'ivory gull',
    'ring billed gull',
    'slaty backed gull',
    'western gull',
    'anna hummingbird',
    'ruby throated hummingbird',
    'rufous hummingbird',
    'green violetear',
    'long tailed jaeger',
    'pomarine jaeger',
    'blue jay',
    'florida jay',
    'green jay',
    'dark eyed junco',
    'tropical kingbird',
    'gray kingbird',
    'belted kingfisher',
    'green kingfisher',
    'pied kingfisher',
    'ringed kingfisher',
    'white breasted kingfisher',
    'red legged kittiwake',
    'horned lark',
    'pacific loon',
    'mallard',
    'western meadowlark',
    'hooded merganser',
    'red breasted merganser',
    'mockingbird',
    'nighthawk',
    'clark nutcracker',
    'white breasted nuthatch',
    'baltimore oriole',
    'hooded oriole',
    'orchard oriole',
    'scott oriole',
    'ovenbird',
    'brown pelican',
    'white pelican',
    'western wood pewee',
    'sayornis',
    'american pipit',
    'whip poor will',
    'horned puffin',
    'common raven',
    'white necked raven',
    'american redstart',
    'geococcyx',
    'loggerhead shrike',
    'great grey shrike',
    'baird sparrow',
    'black throated sparrow',
    'brewer sparrow',
    'chipping sparrow',
    'clay colored sparrow',
    'house sparrow',
    'field sparrow',
    'fox sparrow',
    'grasshopper sparrow',
    'harris sparrow',
    'henslow sparrow',
    'le conte sparrow',
    'lincoln sparrow',
    'nelson sharp tailed sparrow',
    'savannah sparrow',
    'seaside sparrow',
    'song sparrow',
    'tree sparrow',
    'vesper sparrow',
    'white crowned sparrow',
    'white throated sparrow',
    'cape glossy starling',
    'bank swallow',
    'barn swallow',
    'cliff swallow',
    'tree swallow',
    'scarlet tanager',
    'summer tanager',
    'artic tern',
    'black tern',
    'caspian tern',
    'common tern',
    'elegant tern',
    'forsters tern',
    'least tern',
    'green tailed towhee',
    'brown thrasher',
    'sage thrasher',
    'black capped vireo',
    'blue headed vireo',
    'philadelphia vireo',
    'red eyed vireo',
    'warbling vireo',
    'white eyed vireo',
    'yellow throated vireo',
    'bay breasted warbler',
    'black and white warbler',
    'black throated blue warbler',
    'blue winged warbler',
    'canada warbler',
    'cape may warbler',
    'cerulean warbler',
    'chestnut sided warbler',
    'golden winged warbler',
    'hooded warbler',
    'kentucky warbler',
    'magnolia warbler',
    'mourning warbler',
    'myrtle warbler',
    'nashville warbler',
    'orange crowned warbler',
    'palm warbler',
    'pine warbler',
    'prairie warbler',
    'prothonotary warbler',
    'swainson warbler',
    'tennessee warbler',
    'wilson warbler',
    'worm eating warbler',
    'yellow warbler',
    'northern waterthrush',
    'louisiana waterthrush',
    'bohemian waxwing',
    'cedar waxwing',
    'american three toed woodpecker',
    'pileated woodpecker',
    'red bellied woodpecker',
    'red cockaded woodpecker',
    'red headed woodpecker',
    'downy woodpecker',
    'bewick wren',
    'cactus wren',
    'carolina wren',
    'house wren',
    'marsh wren',
    'rock wren',
    'winter wren',
    'common yellowthroat'
]
