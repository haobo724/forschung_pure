import os
import sys
import torch
import logging
import pytorch_lightning as pl

# Model import
from models.BasicUnet import BasicUnet

# Loss import
from loss import CELoss,DiceLoss
import monai
from argparse import ArgumentParser

from data_module.song_dataset import Song_dataset_2d_with_CacheDataloder
import helpers as helpers
from pytorch_lightning.callbacks import ModelCheckpoint

from base_train_2D import BasetRAIN
from pytorch_lightning.loggers import TensorBoardLogger
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append(os.path.dirname(__file__))


# This demo contains the training and test pipeline of a 2D U-Net for organ segmentation.
# All 2D pipelines inherit the 2D base pipeline class. For complete implementation of training
# pipeline pls see base_train_2D.py
class benchmark_unet_2d(BasetRAIN):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.model = BasicUnet(in_channels=1, out_channels=4, nfilters=32).cuda()
        weights = [0.5, 2.0, 1.0, 1.0]

        # self.loss = CELoss(weight=weights)
        # self.loss = monai.losses.DiceLoss(to_onehot_y=True)
        self.loss= DiceLoss(weight=weights)
        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        return parser


# main function
def cli_main():
    pl.seed_everything(1234)
    # Get experiment id
    fname = os.path.splitext(os.path.basename(__file__))[0]

    # parse the arguments
    # All pipelines should use python argparser for configuration, so that training is easier on cluster
    parser = ArgumentParser()
    parser = helpers.add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = benchmark_unet_2d.add_model_specific_args(parser)
    parser.add_argument('--datasetmode',type=int, required=True,help='4 mode',default=1)
    parser.add_argument('--opt',type=str, required=True,help='2 optimizers',default='Adam')
    parser.add_argument('--resume',type=bool, required=False,help='if continue train',default=False)
    parser.add_argument('--lastcheckpoint',type=str, required=False,help='path to lastcheckpoint',default='')
    parser.add_argument('--hpar',type=str, required=False,help='path to lastcheckpoint',default='')
    args = parser.parse_args()
    print(args.resume)
    # --resume
    # False
    # --lastcheckpoint
    # F:\Forschung\pure\lightning_logs\mode3\my_model\version_0\checkpoints\last.ckpt
    # --hpar
    # F:\Forschung\pure\lightning_logs\mode3\my_model\version_0\hparams.yaml
    # create the pipeline

    # Ckpt callbacks
    ckpt_callback = ModelCheckpoint(
        monitor='avg_iousummean',
        save_top_k=2,
        mode='max',
        save_last=True,
        filename='{epoch:02d}-{avg_iousummean:.02f}'
    )
    if not os.path.exists(os.path.join('.', 'lightning_logs', f'mode{args.datasetmode}')):
        os.makedirs(os.path.join('.', 'lightning_logs', f'mode{args.datasetmode}'))

    logger = TensorBoardLogger(save_dir=os.path.join('.', 'lightning_logs', f'mode{args.datasetmode}'), name='my_model')
    # create trainer using pytorch_lightning
    if args.resume:
        print("Resume")
        net = benchmark_unet_2d(hparams=vars(args))
        trainer = pl.Trainer.from_argparse_args(args,precision=16,check_val_every_n_epoch=2,callbacks=[ckpt_callback],num_sanity_val_steps=0,logger=logger
                                                ,resume_from_checkpoint=args.lastcheckpoint)
    else:
        net = benchmark_unet_2d(hparams=vars(args))
        trainer = pl.Trainer.from_argparse_args(args,precision=16,check_val_every_n_epoch=2,callbacks=[ckpt_callback],num_sanity_val_steps=0,logger=logger)

    logging.info(f'Manual logging starts. Model version: {trainer.logger.version}_mode{args.datasetmode}')

    # configure data module
    logging.info(f'dataset from {args.data_folder}')

    dm = Song_dataset_2d_with_CacheDataloder(args.data_folder[0],
                                 worker=args.worker,
                                 batch_size=args.batch_size,
                                 mode=args.datasetmode)

    dm.setup(stage='fit')

    # training
    #
    # lr =trainer.tuner.lr_find(model=net,datamodule=dm)

    # # fig=lr.plot(suggest=True)
    # # fig.show()
    # net.lr=lr.suggestion()
    # print('best initial lr:',net.lr)
    trainer.fit(model=net,datamodule=dm)

    logging.info("!!!!!!!!!!!!!!This is the end of the training!!!!!!!!!!!!!!!!!!!!!!")
    print('THE END')



if __name__ == "__main__":
    cli_main()
    #
    # model_infer(models=glob.glob('.\\lightning_logs\\version_650051\\**\\*.ckpt', recursive=True),
    #             raw_dir='D:\\Data\\ct_data\\visceral_manual_seg\\test',
    #             tar_dir=None,
    #             batch_size=10)

    # organ-wise analysis
    # helpers.MOS_eval(pred_path="D:\\Chang\\MultiOrganSeg\\model_output\\benchmark_unet_2D\\10000081_ct\\10000081_ct_seg.nii.gz",
    #                  gt_path="D:\\Data\\ct_data\\test\\10000081\\GroundTruth.nii.gz")

    # model_debug()
