# 3D-RETR: End-to-End Single and Multi-View3D Reconstruction with Transformers (BMVC 2021)


### **Zai Shi***, **Zhao Meng***, **Yiran Xing**, **Yunpu Ma**, **Roger Wattenhofer**   

∗The first two authors contribute equally to this work

[[Arxiv](https://arxiv.org/abs/2110.08861)]
 
![image](https://user-images.githubusercontent.com/14837467/137624930-96072863-a32b-431f-ab20-985ffd1e51f4.png)

## Create Environment

```
git clone git@github.com:FomalhautB/3D-RETR.git
cd 3D-RETR
conda env create -f config/environment.yaml
conda activate 3d-retr
```

## Prepare Data

### ShapeNet

Download the [Rendered Images](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz) and [Voxelization (32)](http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz) and decompress them into `$SHAPENET_IMAGE` and `$SHAPENET_VOXEL`

## Train

Here is an example of reproducing the result of the single view 3D-RETR-B on the ShapeNet dataset:

```
python train.py \
    --model image2voxel \
    --transformer_config config/3d-retr-b.yaml \
    --annot_path data/ShapeNet.json \
    --model_path $SHAPENET_VOX \
    --image_path $SHAPENET_IMAGES \
    --gpus 1 \
    --precision 16 \
    --deterministic \
    --train_batch_size 16 \
    --val_batch_size 16 \
    --num_workers 4 \
    --check_val_every_n_epoch 1 \
    --accumulate_grad_batches 1 \
    --view_num 1 \
    --sample_batch_num 0 \
    --loss_type dice \
```

