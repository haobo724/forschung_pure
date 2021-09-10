import os
import shutil
import sys
import torch
import monai
from argparse import ArgumentParser
import pytorch_lightning as pl
import helpers
from monai.data.utils import pad_list_data_collate
from ds import climain
from pytorch_lightning.loggers import TensorBoardLogger
import time

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
    if args.ckpt == 'local':
        modelslist = []
        for root, dirs, files in os.walk(r"F:\Forschung\multiorganseg\good\onlyfresh"):
            for file in files:
                if file.endswith('.ckpt'):
                    modelslist.append(os.path.join(root, file))
        args.ckpt = modelslist[0]


    infer_ds, _, _ = climain(args.data_folder[0], Input_worker=4, mode='test', dataset_mode=5,clean=args.clean)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=4,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate

    )
    test = torch.load(args.ckpt)
    datamode = test['hyper_parameters']['datasetmode']
    loss_method = test['hyper_parameters']['loss']
    args.loss=loss_method
    print(args.loss)
    model = benchmark_unet_2d.load_from_checkpoint(args.ckpt, hparams=vars(args))

    start_time = time.time()
    trainer = pl.Trainer(gpus=-1, logger=logger, precision=16)
    trainer.test(model, infer_loader)
    if os.path.exists('saved_images'):
        newname = f'saved_images_mode{datamode}_{loss_method}'
        if os.path.exists(newname):
            shutil.rmtree(newname)
        os.rename('saved_images', newname)
    print('time:', time.time() - start_time)  # time: 91.36657166481018


if __name__ == "__main__":
    # root,dirs,files=os.walk('./mostoolkit/lightning_logs/version_65')

    infer()

    # modelslist=[]
    # for root,dirs,files in os.walk(r"F:\Forschung\multiorganseg\good\onlyfresh"):
    #     for file in files:
    #         if file.endswith('.ckpt'):
    #             modelslist.append(os.path.join(root, file))
    # infer(modelslist[0],r'F:\Forschung\multiorganseg\data\train_2D')
