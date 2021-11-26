import argparse
import os
from pprint import PrettyPrinter

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import ShapeNetDataset, ShuffleDataset, transforms, normalize
from src.data.binvox_rw import Voxels
from src.image2voxel import Image2Voxel
from src.utils import load_config


def save_binvox(voxel, dest, translate, scale):
    binvox = Voxels(voxel, voxel.shape, translate, scale, 'xyz')
    binvox.write(open(dest, 'wb'))


def to_numpy(image):
    image.convert("RGB")
    return [np.asarray(image, dtype=np.float32) / 255]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train transformer conditioned on image inputs')
    parser.add_argument('--annot_path', type=str, required=True,
                        help='Path to "ShapeNet.json"')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the voxel models')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input images')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Manual seed for python, numpy and pytorch')
    parser.add_argument('--split', type=str, default='val',
                        help='"train", "test", or "val"')
    parser.add_argument("--transformer_config", type=str, default=None,
                        help='Path to the image2voxel config file')
    parser.add_argument("--background", type=int, nargs=3, default=(0, 0, 0),
                        help='The (R, G, B) color for the image background')
    parser.add_argument("--beam", type=int, default=1,
                        help='Number of beams for generation')
    parser.add_argument("--view_num", type=int, default=1,
                        help='Number of views for the image input')
    parser.add_argument("--threshold", type=float, default=0.5,
                        help='Threshold for deciding voxel occupancy')
    parser.add_argument("--predict", action='store_true',
                        help='Predict and save results')
    parser.add_argument("--save_path", type=str, default=None,
                        help='Path to save the prediction')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    if args.resume_from_checkpoint is None:
        raise ValueError('No checkpoint specified')

    pp = PrettyPrinter(indent=4)
    pp.pprint(vars(args))

    # =================================================================================

    pl.seed_everything(args.seed)

    image_trans = transforms.Compose([
        to_numpy,
        transforms.CenterCrop((224, 224), (128, 128)),
        transforms.RandomBackground(((240, 240), (240, 240), (240, 240))),
        transforms.ToTensor(),
        lambda x: x[0],
        normalize
    ])

    dataset_params = {
        'annot_path': args.annot_path,
        'model_path': args.model_path,
        'image_path': args.image_path
    }
    dataset = ShapeNetDataset(
        **dataset_params,
        image_transforms=image_trans,
        split=args.split,
        mode='first',
        background=args.background,
        view_num=args.view_num
    )

    dataset = ShuffleDataset(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # =================================================================================

    transformer_config = load_config(args.transformer_config)
    pp.pprint(transformer_config)
    model = Image2Voxel.load_from_checkpoint(
        threshold=args.threshold,
        checkpoint_path=args.resume_from_checkpoint,
        **transformer_config
    )

    trainer = pl.Trainer.from_argparse_args(args, logger=False)

    if args.predict:
        if args.save_path is None:
            raise ValueError('save_path is not specified')

        prediction = trainer.predict(model, loader)
        for pred_dict in tqdm(prediction):
            for i in range(len(pred_dict['generation'])):
                tax_path = os.path.join(args.save_path, pred_dict['taxonomy_id'][i], pred_dict['model_id'][i])

                if not os.path.isdir(tax_path):
                    os.makedirs(tax_path)

                voxel = pred_dict['generation'][i][0].cpu().numpy()
                save_binvox(
                    voxel.astype(np.bool),
                    os.path.join(tax_path, 'prediction.binvox'),
                    pred_dict['translate'][i].cpu().numpy(),
                    pred_dict['scale'][i].cpu().numpy(),
                )
    else:
        trainer.test(model, loader)
