import json
import os
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from .binvox_rw import read_as_3d_array

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ShuffleDataset(Dataset):
    def __init__(self, dataset):
        super(ShuffleDataset, self).__init__()
        self._dataset = dataset
        self._indices = torch.randperm(len(dataset))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[self._indices[index].item()]


class ShapeNetDataset(Dataset):
    def __init__(
            self,
            annot_path: str,
            model_path: str,
            image_path: str,
            image_transforms: Callable,
            split: str = 'train',
            mode: str = 'random',
            background=(255, 255, 255),
            view_num: int = 1
    ):
        """
        @param annot_path: path to the "ShapeNet.json" file
        @param model_path: path to the "ShapeNetVox32" folder
        @param image_path: path to the "ShapeNetRendering" folder
        @param image_transforms: preprocessing transformations for images
        @param split: one of "train", "val", "test"
        @param mode:
            - random: load a random image if there are multiple
            - first: always load the first images
        @param view_num: load view_num of images at once
            - >= 1: image size: view * c * h * w
            - -1: image size: c * h * w
        """

        if split not in ['train', 'val', 'test']:
            raise ValueError('Unexpected split')

        if mode not in ['random', 'first']:
            raise ValueError('Unexpected mode')

        with open(annot_path) as annot_file:
            annots = json.load(annot_file)

        self._meta_data = []
        for taxonomy in annots:
            for model_id in taxonomy[split]:
                self._meta_data.append({
                    'taxonomy_id': taxonomy['taxonomy_id'],
                    'taxonomy_name': taxonomy['taxonomy_name'],
                    'model_id': model_id
                })

        self._model_path = model_path
        self._image_path = image_path
        self._image_transforms = image_transforms
        self._mode = mode
        self._background = background
        self._view_num = view_num

    def __getitem__(self, index):
        meta_data = self._meta_data[index]
        taxonomy_id = meta_data['taxonomy_id']
        model_id = meta_data['model_id']

        binvox_path = os.path.join(self._model_path, taxonomy_id, model_id, 'model.binvox')

        with open(binvox_path, 'rb') as f:
            raw_voxel = read_as_3d_array(f)
            voxel = raw_voxel.data.astype(np.float32)

        image_base_path = os.path.join(self._image_path, taxonomy_id, model_id, 'rendering')
        image_file_list = list(os.listdir(image_base_path))
        image_file_list.sort()
        image_file_list.remove('rendering_metadata.txt')
        image_file_list.remove('renderings.txt')

        if self._mode == 'random':
            image_indices = torch.randperm(len(image_file_list))
        else:
            image_indices = torch.arange(len(image_file_list))

        image_indices = image_indices[:self._view_num]

        images = []
        for image_index in image_indices:
            image_path = os.path.join(image_base_path, image_file_list[image_index.item()])
            rgba = Image.open(image_path)
            image = Image.new("RGB", rgba.size, self._background)
            image.paste(rgba, mask=rgba.split()[3])
            image = self._image_transforms(image)
            images.append(image)

        images = torch.stack(images)
        if self._view_num == 1:
            images = images.squeeze(0)

        return {
            'image': images,
            'id': index,
            'model_id': model_id,
            'taxonomy_id': taxonomy_id,
            'taxonomy_name': meta_data['taxonomy_name'],
            'scale': raw_voxel.scale,
            'translate': torch.Tensor(raw_voxel.translate),
            'voxel': torch.Tensor(voxel).unsqueeze(0)
        }

    def __len__(self):
        return len(self._meta_data)
