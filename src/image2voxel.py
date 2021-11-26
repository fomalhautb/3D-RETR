import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.optim import AdamW

from src.build import build_voxel_decoder
from src.data import denormalize
from src.models import VisionTransformerEncoder
from src.utils import compute_iou, plot_voxels


class Image2Voxel(pl.LightningModule):
    def __init__(
            self,
            encoder_dropout=0.3,
            encoder_model='vit_deit_base_distilled_patch16_224',
            decoder_model='vae',
            decoder_dropout=0.3,
            decoder_depth=6,
            decoder_heads=6,
            decoder_dim=768,
            sample_batch_num=0,
            patch_num=4,
            voxel_size=32,
            sched_factor=1,
            lr=1e-4,
            beam=1,
            num_resnet_blocks=2,
            cnn_hidden_dim=64,
            num_cnn_layers=3,
            threshold=0.5,
            loss_type='dice',
            pretrained=True,
    ):
        super(Image2Voxel, self).__init__()
        self.save_hyperparameters()

        self.encoder = VisionTransformerEncoder(
            attn_dropout=encoder_dropout,
            model=encoder_model,
            pretrained=pretrained
        )

        self.decoder = build_voxel_decoder(
            decoder_model=decoder_model,
            decoder_dropout=decoder_dropout,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            decoder_dim=decoder_dim,
            patch_num=patch_num,
            voxel_size=voxel_size,
            num_resnet_blocks=num_resnet_blocks,
            cnn_hidden_dim=cnn_hidden_dim,
            num_cnn_layers=num_cnn_layers
        )

        self.sample_batch_num = sample_batch_num
        self.lr = lr
        self.sched_factor = sched_factor
        self.beam = beam
        self.threshold = threshold
        self.loss_type = loss_type

    def _polarize(self, data):
        data[data > self.threshold] = 1
        data[data <= self.threshold] = 0
        return data

    def generate(self, batch, temperature=1.0, sample=False, beam=1):
        image = batch['image']
        context = self._encode(image)
        out = self.decoder.generate(
            context=context,
            temperature=temperature,
            sample=sample,
            beam=beam
        )

        return out

    def configure_optimizers(self):
        opt = AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, self.sched_factor)

        return [opt], [{'scheduler': sched, 'interval': 'step'}]

    def _encode(self, image):
        if len(image.size()) == 4:
            return self.encoder(image)
        else:
            batch_size, view_count = image.size(0), image.size(1)
            image = rearrange(image, 'b v c h w -> (b v) c h w')
            context = self.encoder(image)
            context = rearrange(context, '(b v) l d -> b v l d', b=batch_size, v=view_count)
            context = context.mean(dim=1)
            return context

    def training_step(self, batch, batch_idx):
        voxel, image = batch['voxel'], batch['image']
        context = self._encode(image)
        loss = self.decoder.get_loss(x=voxel, context=context, loss_type=self.loss_type)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        gens = self.generate(batch, beam=self.beam)
        gens = self._polarize(gens)

        ious = []
        for gen, gt in zip(gens, batch['voxel']):
            ious.append(compute_iou(gen, gt).item())

        self.log('val_iou', sum(ious) / len(ious))

        if batch_idx < self.sample_batch_num:
            for i in range(len(gens)):
                pred_img = plot_voxels(gens[i], rot02=1, rot12=1).convert('RGB')
                gt_img = plot_voxels(batch['voxel'][i], rot02=1, rot12=1).convert('RGB')
                img_id = str(batch['id'][i].item())
                input_img = np.moveaxis(batch['image'][i].cpu().numpy(), 0, -1)
                input_img = denormalize(input_img)
                self.logger.experiment.log_image(
                    self.logger.run_id,
                    input_img,
                    f'step{self.global_step}/{img_id}image.jpg'
                )
                self.logger.experiment.log_image(
                    self.logger.run_id,
                    pred_img,
                    f'step{self.global_step}/{img_id}pred.jpg'
                )
                self.logger.experiment.log_image(
                    self.logger.run_id,
                    gt_img,
                    f'step{self.global_step}/{img_id}gt.jpg'
                )

        return ious, batch['taxonomy_name']

    def test_step(self, batch, batch_idx):
        gens = self.generate(batch, beam=self.beam)
        gens = self._polarize(gens)

        ious = []
        for gen, gt in zip(gens, batch['voxel']):
            ious.append(compute_iou(gen, gt).item())

        self.log('test_iou', sum(ious) / len(ious))

        return ious, batch['taxonomy_name']

    def predict_step(self, batch, batch_idx, dataloader_idx):
        gens = self.generate(batch, beam=self.beam)
        gens = self._polarize(gens)

        return {
            'generation': gens,
            'scale': batch['scale'],
            'translate': batch['translate'],
            'model_id': batch['model_id'] if 'model_id' in batch else batch['id'],
            'taxonomy_name': batch['taxonomy_name'],
            'taxonomy_id': batch['taxonomy_id']
        }

    def test_epoch_end(self, step_outputs):
        iou_dict = self.get_class_iou(step_outputs)

        for tax, iou in iou_dict.items():
            self.log(f'test_iou_{tax}', iou)

    def validation_epoch_end(self, step_outputs):
        iou_dict = self.get_class_iou(step_outputs)

        for tax, iou in iou_dict.items():
            self.log(f'val_iou_{tax}', iou)

    def get_class_iou(self, step_outputs):
        # flatten step outputs
        tax_list = [tax for output in step_outputs for tax in output[1]]
        iou_list = [iou for output in step_outputs for iou in output[0]]

        iou_dict = {}

        for name in set(tax_list):
            # filter out ious for each class
            ious = [iou for iou, tax in zip(iou_list, tax_list) if tax == name]
            iou_dict[name] = sum(ious) / len(ious)

        iou_dict['mean'] = sum(list(iou_dict.values())) / len(iou_dict)

        return iou_dict
