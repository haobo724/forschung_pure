import os
import sys
import torch
import monai
from argparse import ArgumentParser
import pytorch_lightning as pl
import helpers
from monai.data.utils import pad_list_data_collate
from ds import climain
from pytorch_lightning.loggers import TensorBoardLogger

torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append(os.path.dirname(__file__))

from basetrain_song import benchmark_unet_2d

def infer():

    parser = ArgumentParser()
    parser = helpers.add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = benchmark_unet_2d.add_model_specific_args(parser)
    args = parser.parse_args()
    logger = TensorBoardLogger(save_dir=os.path.join('.', 'lightning_logs', f'mode{args.datasetmode}'), name='my_test')
    if args.ckpt=='local':
        modelslist=[]
        for root,dirs,files in os.walk(r"F:\Forschung\multiorganseg\good\onlyfresh"):
            for file in files:
                if file.endswith('.ckpt'):
                    modelslist.append(os.path.join(root, file))
        args.ckpt=modelslist[0]
    # images = sorted(glob.glob(os.path.join(raw_dir, '*_ct.nii.gz')))
    # labels = sorted(glob.glob(os.path.join(raw_dir, '*_seg.nii.gz')))
    # assert len(images) == len(labels)
    # keys = ("image", "label","leaky")
    #
    # val_imgs = [
    #     {keys[0]: img, keys[1]: seg, keys[2]:seg} for img, seg in
    #     zip(images[350:360], labels[350:360])
    # ]
    # infer_xform = Song_dataset_2d_with_CacheDataloder.get_xform(mode="infer")
    # infer_ds = monai.data.CacheDataset(data=val_imgs, transform=infer_xform)

    infer_ds,_,_=climain(args.data_folder[0],Input_worker=4,mode='test',dataset_mode=5)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=8,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate
    )
    model = benchmark_unet_2d.load_from_checkpoint(args.ckpt,hparams=vars(args))
    trainer=pl.Trainer(gpus=-1,logger=logger)
    trainer.test(model,infer_loader)

if __name__ == "__main__":
    # root,dirs,files=os.walk('./mostoolkit/lightning_logs/version_65')

    infer()

    # modelslist=[]
    # for root,dirs,files in os.walk(r"F:\Forschung\multiorganseg\good\onlyfresh"):
    #     for file in files:
    #         if file.endswith('.ckpt'):
    #             modelslist.append(os.path.join(root, file))
    # infer(modelslist[0],r'F:\Forschung\multiorganseg\data\train_2D')
