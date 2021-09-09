import os
import sys
import torch
import logging
import pytorch_lightning as pl

# Model import
from models.BasicUnet import BasicUnet
from models.Unet_song import UNET

# Loss import
from loss import CELoss
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
    '''

    CEloss target不用onehot，也就是Dice的target就是一层就好
    Dice需要，也就是和prediction同size，并且根据公式应该对prediction做sigmod

    '''

    def __init__(self, hparams):
        super().__init__(hparams)

        # self.model = BasicUnet(in_channels=1, out_channels=4, nfilters=32).cuda()
        self.model = UNET(in_channels=1, out_channels=4).cuda()
        Loss_weights = [0.5, 1.0, 1.0, 1.0]
        if hparams['loss'] == 'CE':
            self.loss = CELoss(weight=Loss_weights)
            print("CELoss will be used")
        else:
            self.loss = monai.losses.DiceLoss(to_onehot_y=True)

            # self.loss =monai.losses.FocalLoss(gamma=2,to_onehot_y=True)

            # self.loss= DiceLoss(weight=weights)
            print("DiceLoss will be used")

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--loss', type=str, default='CE')
        parser.add_argument('--clean', type=bool, default=False)
        return parser


class ContinueTrain(benchmark_unet_2d):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.model = benchmark_unet_2d.load_from_checkpoint(hparams['lastcheckpoint'])
        # self.model.freeze()

        self.save_hyperparameters()
    def forward(self, x):
        return self.model(x)



# main function
def cli_main():
    pl.seed_everything(1234)
    # Get experiment id

    # parse the arguments
    # All pipelines should use python argparser for configuration, so that training is easier on cluster
    parser = ArgumentParser()
    parser = helpers.add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = benchmark_unet_2d.add_model_specific_args(parser)

    args = parser.parse_args()
    print('resume:', args.resume)
    # --resume
    # False
    # --lastcheckpoint
    # F:\Forschung\pure\lightning_logs\mode6\my_model\version_157\checkpoints\last.ckpt
    # --hpar
    # F:\Forschung\pure\lightning_logs\mode6\my_model\version_157\hparams.yaml
    # create the pipeline

    # Ckpt callbacks
    ckpt_callback = ModelCheckpoint(
        # monitor='valid_loss',
        monitor='avg_iousummean',
        save_top_k=2,
        mode='max',
        save_last=True,
        filename='{epoch:02d}-{valid_loss:.02f}'
    )
    saved_path=os.path.join('.', 'lightning_logs', f'mode{args.datasetmode}',
                            f'{args.loss}_'+f'clean_{args.clean}_'+f'resume_{args.resume}')

    logger = TensorBoardLogger(save_dir=saved_path, name='my_model')
    # create trainer using pytorch_lightning
    if args.resume:
        print("Resume")
        # net = benchmark_unet_2d(hparams=vars(args)).load_from_checkpoint(args.lastcheckpoint)
        net = ContinueTrain(hparams=vars(args))
        trainer = pl.Trainer.from_argparse_args(args, precision=16, check_val_every_n_epoch=2,
                                                callbacks=[ckpt_callback], num_sanity_val_steps=0, logger=logger
                                                )
    else:
        net = benchmark_unet_2d(hparams=vars(args))
        trainer = pl.Trainer.from_argparse_args(args, precision=16, check_val_every_n_epoch=2,
                                                callbacks=[ckpt_callback], num_sanity_val_steps=0, logger=logger)

    logging.info(f'Manual logging starts. Model version: {trainer.logger.version}_mode{args.datasetmode}')

    # configure data module
    logging.info(f'dataset from {args.data_folder}')

    dm = Song_dataset_2d_with_CacheDataloder(args.data_folder[0],
                                             worker=0,
                                             batch_size=args.batch_size,
                                             mode=args.datasetmode,
                                             clean=args.clean)

    # dm.setup(stage='fit')

    # training
    #
    # lr =trainer.tuner.lr_find(model=net,datamodule=dm)

    # # fig=lr.plot(suggest=True)
    # # fig.show()
    # net.lr=lr.suggestion()
    # print('best initial lr:',net.lr)
    trainer.fit(model=net, datamodule=dm)

    logging.info("!!!!!!!!!!!!!!This is the end of the training!!!!!!!!!!!!!!!!!!!!!!")
    print('THE END')
    sys.exit()


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
