import os
import sys

import matplotlib.pyplot as plt
import torch
import logging
import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import dice_score

torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append(os.path.dirname(__file__))

class BasetRAIN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = None
        self.loss = None
        self.hparamss = hparams
        self.train_logger = logging.getLogger(__name__)
        self.validation_recall = pl.metrics.Recall(average='macro', mdmc_average='samplewise', num_classes=4)
        self.validation_precision = pl.metrics.Precision(average='macro', mdmc_average='samplewise', num_classes=4)
        self.validation_IOU = pl.metrics.IoU( num_classes=4,absent_score=True)
        self.validation_IOU2 = pl.metrics.IoU( num_classes=4,absent_score=True,reduction='none')
        if self.hparamss['datasetmode']==4 :
            self.modifiy_label_ON=True
            print(f'[INFO] modifiy_label_ON={self.modifiy_label_ON}')
        else:
            self.modifiy_label_ON=False
            print(f'[INFO] modifiy_label_ON={self.modifiy_label_ON}')



    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataset_idx=None):
        x, y = batch["image"], batch["label"]
        z_bactch =batch["leaky"]
        y_hat = self(x)
        y_copy = y.clone()
        if self.modifiy_label_ON:
            for idx,z in enumerate(z_bactch):
                if z != 0:
                    # pred = torch.squeeze(y_hat)
                    pred = y_hat.permute(0,2, 3, 1)
                    pred = torch.softmax(pred[idx,...], dim=-1)
                    picked_channel = pred[idx,...].argmax(dim=-1)
                    cords = np.argwhere(picked_channel.cpu().numpy() == z)
                    realcord = []
                    for cord in cords:
                        if y_copy[idx,0,cord[0], cord[1]] == 0:
                            realcord.append(cord)

                    for cord in realcord:
                        y_copy[idx,0,cord[0], cord[1]] = z
                    #todo:如果是肺，右肺（label=3）再来一遍
                    if z ==2:
                        cord_zusatz=np.argwhere(picked_channel.cpu().numpy() == z+1)
                        realcord = []
                        for cord in cord_zusatz:
                            if y_copy[idx, 0, cord[0], cord[1]] == 0:
                                realcord.append(cord)

                        for cord in realcord:
                            y_copy[idx, 0, cord[0], cord[1]] = z+1

        loss = self.loss.forward(y_hat, y_copy)
        self.log("loss", loss, on_step=False,on_epoch=True)
        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        
        def reshape_valid(var):
            # Make sure the val dataset fits the subsampling path. For Unet with depth 5, the factor is 32
            s = var.cpu().numpy().shape
            return torch.nn.functional.interpolate(var, [int(np.ceil(i / 32) * 32) for i in s if i > 1])
        
        def scale_zero_one(var):
            return (var - var.min()) / (var.max() - var.min() + 1e-5)

        x, y = batch["image"], batch["label"]
        # shape = [batch, channel, w, h]
        z_bactch= batch["leaky"]
        pred = self(x)
        y_copy = y.clone()
        if self.modifiy_label_ON:
            for idx, z in enumerate(z_bactch):
                if z != 0:
                    # pred = torch.squeeze(y_hat)
                    predc = pred.permute(0, 2, 3, 1)
                    predc = torch.softmax(predc[idx, ...], dim=-1)
                    picked_channel = predc[idx, ...].argmax(dim=-1)
                    cords = np.argwhere(picked_channel.cpu().numpy() == z)
                    realcord = []
                    for cord in cords:
                        if y_copy[idx, 0, cord[0], cord[1]] == 0:
                            realcord.append(cord)

                    for cord in realcord:
                        y_copy[idx, 0, cord[0], cord[1]] = z
                    # todo:如果是肺，右肺（label=3）再来一遍
                    if z == 2:
                        cord_zusatz = np.argwhere(picked_channel.cpu().numpy() == z + 1)
                        realcord = []
                        for cord in cord_zusatz:
                            if y_copy[idx, 0, cord[0], cord[1]] == 0:
                                realcord.append(cord)

                        for cord in realcord:
                            y_copy[idx, 0, cord[0], cord[1]] = z + 1
                #

        loss = self.loss.forward(pred, y_copy).cpu()

        # argmax
        recall = self.validation_recall(torch.nn.functional.softmax(pred, dim=1), y_copy.long())
        precision = self.validation_precision(torch.nn.functional.softmax(pred, dim=1), y_copy.long())
        iou = self.validation_IOU(torch.nn.functional.softmax(pred, dim=1), y_copy.long())
        iou_individual = self.validation_IOU2(torch.nn.functional.softmax(pred, dim=1), y_copy.long()).float()
        dice_individual=dice_score(torch.nn.functional.softmax(pred, dim=1),y.squeeze(1).long(),reduction='none',bg=True,no_fg_score=1)[:4].float()


        self.log("recall", loss, on_step=False,on_epoch=True,prog_bar=True)
        self.log("precision", precision, on_step=False,on_epoch=True,prog_bar=True)
        self.log("iou", iou, on_step=False,on_epoch=True,prog_bar=True)

        # pred = torch.argmax(pred, dim=1).unsqueeze(1)
        # if batch_idx == 0:
        #     self.logger.experiment.add_image('valid/pred_0', scale_zero_one(pred[0]), dataformats='CHW', global_step=self.global_step)
        #     self.logger.experiment.add_image('valid/ori_0', x[0], dataformats='CHW', global_step=self.global_step)
        #     self.logger.experiment.add_image('valid/gt_0', scale_zero_one(y_copy[0]), dataformats='CHW', global_step=self.global_step)
        #
        #     self.logger.experiment.add_image('valid/pred_1', scale_zero_one(pred[1]), dataformats='CHW', global_step=self.global_step)
        #     self.logger.experiment.add_image('valid/ori_1', x[1], dataformats='CHW', global_step=self.global_step)
        #     self.logger.experiment.add_image('valid/gt_1', scale_zero_one(y_copy[1]), dataformats='CHW', global_step=self.global_step)
        #
        #     self.logger.experiment.add_image('valid/pred_2', scale_zero_one(pred[2]), dataformats='CHW', global_step=self.global_step)
        #     self.logger.experiment.add_image('valid/ori_2', x[2], dataformats='CHW', global_step=self.global_step)
        #     self.logger.experiment.add_image('valid/gt_2', scale_zero_one(y_copy[2]), dataformats='CHW', global_step=self.global_step)
        returndic={}
        returndic.setdefault("loss",loss)
        returndic.setdefault("recall",recall)
        returndic.setdefault("precision",precision)
        returndic.setdefault("iou",iou)
        returndic.setdefault("iou_individual_bg",iou_individual[0])
        returndic.setdefault("iou_individual_liver",iou_individual[1])
        returndic.setdefault("iou_individual_left_lung",iou_individual[2])
        returndic.setdefault("iou_individual_right_lung",iou_individual[3])

        returndic.setdefault("dice_individual_bg", dice_individual[0])
        returndic.setdefault("dice_individual_liver", dice_individual[1])
        returndic.setdefault("dice_individual_left_lung", dice_individual[2])
        returndic.setdefault("dice_individual_right_lung", dice_individual[3])

        return returndic

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.train_logger.info("Training epoch {} ends".format(self.current_epoch))
        self.log('train/loss', avg_loss)
    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recall = torch.stack([x['recall'] for x in outputs]).mean()
        avg_precision = torch.stack([x['precision'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        avg_iou_individual_bg = torch.stack([x['iou_individual_bg'] for x in outputs]).mean()
        avg_iou_individual_liver = torch.stack([x['iou_individual_liver'] for x in outputs]).mean()
        avg_iou_individual_left_lung = torch.stack([x['iou_individual_left_lung'] for x in outputs]).mean()
        avg_iou_individual_right_lung = torch.stack([x['iou_individual_right_lung'] for x in outputs]).mean()

        avg_dice_individual_bg = torch.stack([x['dice_individual_bg'] for x in outputs]).mean()
        avg_dice_individual_liver = torch.stack([x['dice_individual_liver'] for x in outputs]).mean()
        avg_dice_individual_left_lung = torch.stack([x['dice_individual_left_lung'] for x in outputs]).mean()
        avg_dice_individual_right_lung = torch.stack([x['dice_individual_right_lung'] for x in outputs]).mean()


        self.train_logger.info(f"Validatoin epoch {self.current_epoch} ends, val_loss = {avg_loss}")
        self.log('valid/loss', avg_loss)
        self.log('valid/recall', avg_recall)
        self.log('valid/precision', avg_precision)
        self.log('valid/IOU', avg_iou)
        self.log('valid/avg_iou_individual_bg', avg_iou_individual_bg)
        self.log('valid/avg_iou_individual_liver', avg_iou_individual_liver)
        self.log('valid/avg_iou_individual_left_lung', avg_iou_individual_left_lung)
        self.log('valid/avg_iou_individual_right_lung', avg_iou_individual_right_lung)
        self.log('valid/avg_dice_individual_bg', avg_dice_individual_bg)
        self.log('valid/avg_dice_individual_liver', avg_dice_individual_liver)
        self.log('valid/avg_dice_individual_left_lung', avg_dice_individual_left_lung)
        self.log('valid/avg_dice_individual_right_lung', avg_dice_individual_right_lung)

    def configure_optimizers(self):
        if self.hparamss['opt']=='Adam':
            print('[INFO] Adam will be used')
            return torch.optim.Adam(self.parameters(), lr=self.hparamss['lr'])
        else:
            print('[INFO] SGD will be used')
            return torch.optim.SGD(self.parameters(), lr=self.hparamss['lr'])

    def test_step(self, batch, batch_idx, dataset_idx=None):
        x, y = batch['image'], batch['label']
        y_hat = self(x)
        iou = self.validation_IOU(torch.nn.functional.softmax(y_hat, dim=1), y.long())
        iou2 = self.validation_IOU2(torch.nn.functional.softmax(y_hat, dim=1), y.long())
        print(iou2)
        # pred=torch.squeeze(y_hat)
        # pred=pred.permute(1,2,0)
        pred=torch.softmax(y_hat,dim=1)
        picked_channel=pred.argmax(dim=1)

        # print(torch.nn.functional.softmax(y_hat, dim=1).argmax(dim=1))
        dice=dice_score(torch.nn.functional.softmax(y_hat, dim=1),y.squeeze(1).long(),reduction='none',bg=True,no_fg_score=1)[:4]
        print(dice)
        for index in range(x.shape[0]):
            fig, axs = plt.subplots(1,4)

            axs[0].imshow(picked_channel[index,...]*0.5+x[index,0,...]*0.5)
            axs[1].imshow(picked_channel[index,...])
            axs[2].imshow(x[index,0,...])
            axs[3].imshow(y[index,0,...])
            axs[0].set_title('blend')
            axs[1].set_title('result')
            axs[2].set_title('img')
            axs[3].set_title('ground truth')
            plt.show()

        loss = self.loss.forward(y_hat, y)
        self.log('test/loss', loss)
        self.log('test/iou', iou)
        return {'loss': loss,"iou": iou}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_iou = torch.stack([x['iou'] for x in outputs]).mean()
        self.train_logger.info("test epoch {} ends, val_loss = {},iou={}".format(self.current_epoch, avg_loss,avg_iou))





if __name__ == "__main__":
    pass
    # model_infer(models=['.\\lightning_logs\\version_650048\\final.ckpt',
    #                     '.\\lightning_logs\\version_650048\\checkpoints\\epoch=10-val_loss=0.00.ckpt',
    #                     '.\\lightning_logs\\version_650048\\checkpoints\\epoch=12-val_loss=0.00.ckpt'],
    #             raw_dir='D:\\Data\\ct_data\\test',
    #             tar_dir=None,
    #             batch_size=10)
    # model_infer()
    
    # organ-wise analysis
    # helpers.MOS_eval(pred_path="D:\\Chang\\MultiOrganSeg\\model_output\\benchmark_unet_2D\\10000081_ct\\10000081_ct_seg.nii.gz",
    #                  gt_path="D:\\Data\\ct_data\\test\\10000081\\GroundTruth.nii.gz")

    # helpers.MOS_eval(pred_path='D:\\Chang\\MultiOrganSeg\\model_output\\benchmark_unet_2d_version_650048\\final\\10000081_ct.nii.gz',
    #                  gt_path="D:\\Data\\ct_data\\test\\10000081\\GroundTruth.nii.gz")
    # helpers.MOS_eval(pred_path='D:\\Chang\\MultiOrganSeg\\model_output\\benchmark_unet_2d_version_650048\\epoch=10-val_loss=0.00\\10000081_ct.nii.gz',
    #                  gt_path="D:\\Data\\ct_data\\test\\10000081\\GroundTruth.nii.gz")
    # helpers.MOS_eval(pred_path='D:\\Chang\\MultiOrganSeg\\model_output\\benchmark_unet_2d_version_650048\\epoch=12-val_loss=0.00\\10000081_ct.nii.gz',
    #                  gt_path="D:\\Data\\ct_data\\test\\10000081\\GroundTruth.nii.gz")
    # model_debug()