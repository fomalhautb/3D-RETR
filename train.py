import argparse
from pprint import PrettyPrinter

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.data import (
    ShapeNetDataset,
    ShuffleDataset,
    normalize,
    transforms
)
from src.image2voxel import Image2Voxel
from src.utils import load_config, get_mlflow_tags


def to_numpy(image):
    image.convert("RGB")
    return [np.asarray(image, dtype=np.float32) / 255]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train transformer conditioned on image inputs')
    parser.add_argument('--annot_path', type=str, required=True,
                        help='Path to the "ShapeNet.json"')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the voxel models')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input images')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for dataloader')
    parser.add_argument('--seed', type=int, default=0,
                        help='Manual seed for python, numpy and pytorch')
    parser.add_argument("--transformer_config", type=str, default=None,
                        help='Path to the image2voxel config file')
    parser.add_argument("--sample_batch_num", type=int, default=0,
                        help='The number of batches to show as example in the logger')
    parser.add_argument("--background", type=int, nargs=3, default=(0, 0, 0),
                        help='The (R, G, B) color for the image background')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument("--sched_factor", type=float, default=1,
                        help='Multiplication factor each training step for the scheduler')
    parser.add_argument("--view_num", type=int, default=1,
                        help='Number of views for the image input')
    parser.add_argument("--threshold", type=float, default=0.5,
                        help='Threshold for deciding voxel occupancy')
    parser.add_argument("--data_aug", action='store_true',
                        help='use data augmentation')
    parser.add_argument("--loss_type", type=str, default='dice',
                        help='Loss function type ("dice", "ce", "ce_dice", "focal")')
    parser.add_argument("--experiment_name", type=str, default='3D-RETR',
                        help='Experiment name for mlflow.')

    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

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

    train_dataset = ShapeNetDataset(
        **dataset_params,
        image_transforms=image_trans,
        split='train',
        background=args.background,
        view_num=args.view_num
    )

    val_dataset = ShapeNetDataset(
        **dataset_params,
        image_transforms=image_trans,
        split='val',
        mode='first',
        background=args.background,
        view_num=args.view_num
    )

    val_dataset = ShuffleDataset(val_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    # =================================================================================

    transformer_config = load_config(args.transformer_config)
    pp.pprint(transformer_config)
    model = Image2Voxel(
        sample_batch_num=args.sample_batch_num,
        lr=args.lr,
        sched_factor=args.sched_factor,
        threshold=args.threshold,
        loss_type=args.loss_type,
        **transformer_config
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_iou_mean',
        filename='{epoch:02d}-iou{val_iou:.5f}',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    mlf_logger = pl.loggers.MLFlowLogger(
        experiment_name=args.experiment_name,
        tags=get_mlflow_tags()
    )
    trainer = pl.Trainer.from_argparse_args(args, logger=mlf_logger, callbacks=[checkpoint_callback])
    trainer.logger.log_hyperparams(model.hparams_initial)
    trainer.logger.log_hyperparams({'command_line': vars(args)})
    trainer.fit(model, train_loader, val_loader)
