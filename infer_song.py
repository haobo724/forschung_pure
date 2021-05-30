import os
import sys
import torch
import monai
import numpy as np
import pytorch_lightning as pl
from data_module.song_dataset import Visceral_dataset_2d

import glob

torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append(os.path.dirname(__file__))

from basetrain_song import benchmark_unet_2d

def infer(models, raw_dir):
    if raw_dir is None or models is None:
        ValueError('raw_dir or model is missing')
        
    images = sorted(glob.glob(os.path.join(raw_dir, '*_ct.nii.gz')))
    labels = sorted(glob.glob(os.path.join(raw_dir, '*_seg.nii.gz')))
    assert len(images) == len(labels)
    keys = ("image", "label")

    val_imgs = [
        {keys[0]: img, keys[1]: seg} for img, seg in
        zip(images[350:360], labels[350:360])
    ]
    infer_xform = Visceral_dataset_2d.get_xform(mode="val")
    infer_ds = monai.data.CacheDataset(data=val_imgs, transform=infer_xform)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,
        num_workers=2
    )
    model = benchmark_unet_2d.load_from_checkpoint(models)
    trainer=pl.Trainer()
    trainer.test(model,infer_loader)

if __name__ == "__main__":
    # root,dirs,files=os.walk('./mostoolkit/lightning_logs/version_65')
    # print(root)
    modelslist=[]
    for root,dirs,files in os.walk(r"F:\Forschung\multiorganseg\good"):
        for file in files:
            if file.endswith('.ckpt'):
                modelslist.append(os.path.join(root, file))
    print(modelslist)
    print(modelslist[3])
    infer(modelslist[3],'../data/train_2D')
