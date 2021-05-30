import os
import sys
import torch
import logging
import pytorch_lightning as pl
import shutil

# Model import
from models.BasicUnet import BasicUnet

# Loss import
from loss import CELoss

from argparse import ArgumentParser

from data_module.song_dataset import Visceral_dataset_2d
from data_module.song_dataset import Song_dataset_2d_with_CacheDataloder
import helpers as helpers
from pytorch_lightning.callbacks import ModelCheckpoint

from base_train_2D import BasetRAIN

torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append(os.path.dirname(__file__))


# This demo contains the training and test pipeline of a 2D U-Net for organ segmentation.
# All 2D pipelines inherit the 2D base pipeline class. For complete implementation of training
# pipeline pls see base_train_2D.py
class benchmark_unet_2d(BasetRAIN):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.model = BasicUnet(in_channels=1, out_channels=4, nfilters=32).cuda()
        weights = [0.5, 1.0, 3.0, 5.0]

        self.loss = CELoss(weight=weights)

        self.hparamss = hparams
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
    parser.add_argument('--datasetmethod',type=int, help='1:PersistentDataset other:CacheDataset')
    parser.add_argument('--datasetmode',type=int, required=True,help='4 mode',default=1)
    args = parser.parse_args()

    # create the pipeline
    net = benchmark_unet_2d(hparams=vars(args))

    # Ckpt callbacks
    ckpt_callback = ModelCheckpoint(
        # dirpath='.\\lightning_logs\\debug_network',
        monitor='valid/loss',
        save_top_k=2,
        save_last=True,
        mode='min',
        filename='{epoch:02d}-{val_loss:.2f}'
    )

    # create trainer using pytorch_lightning
    trainer = pl.Trainer.from_argparse_args(args, callbacks=[ckpt_callback])

    # make the direcrory for the checkpoints
    if not os.path.exists(os.path.join('.', 'lightning_logs', f'version_{trainer.logger.version}')):
        os.makedirs(os.path.join('.', 'lightning_logs', f'version_{trainer.logger.version}'))

    # configuration of event log
    helpers.logging_init(log_fname=os.path.join('.', 'lightning_logs',
                                                f'version_{trainer.logger.version}',
                                                f'{fname}.log'),
                         log_lvl=logging.INFO)
    logging.info(f'Manual logging starts. Model version: {trainer.logger.version}')

    # configure data module
    logging.info(f'dataset from {args.data_folder}')
    if os.path.exists(args.cache_dir):
        shutil.rmtree(args.cache_dir)
        logging.info("cache cleaned")
        os.makedirs(args.cache_dir)
        logging.info("cache path rebuilded")


    if args.datasetmethod==1:
        logging.info("PersistentDataset will be used")
        dm = Visceral_dataset_2d(args.data_folder[0],
                             worker=args.worker,
                             batch_size=args.batch_size,
                             cache_dir=args.cache_dir)
    else:
        logging.info("CacheDataset will be used")

        dm = Song_dataset_2d_with_CacheDataloder(args.data_folder[0],
                                 worker=args.worker,
                                 batch_size=args.batch_size,
                                 mode=args.datasetmode)

    dm.setup(stage='train')

    # training
    trainer.fit(net, dm)
    trainer.save_checkpoint(
        os.path.join('.', 'lightning_logs', f'version_{trainer.logger.version}', 'final.ckpt'))
    logging.info("!!!!!!!!!!!!!!This is the end of the training!!!!!!!!!!!!!!!!!!!!!!")
    print('THE END')




def model_debug():
    parser = ArgumentParser()
    parser = helpers.add_training_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = benchmark_unet_2d.add_model_specific_args(parser)
    args = parser.parse_args()

    # log configures

    net = benchmark_unet_2d(hparams=vars(args))
    dummy_x = torch.rand(2, 1, 512, 512).cuda()
    dummy_pred = net(dummy_x)
    print(dummy_pred.shape)

    dummy_y = torch.randint(0, 12, (2, 1, 512, 512)).cuda()
    print('dummmy loss:', net.loss.forward(dummy_pred, dummy_y))


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