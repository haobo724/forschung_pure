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
        self.hparams.update(hparams)
        self.weights = torch.tensor([0.1, 2.0, 1.0, 1.0])
        self.lr=hparams['lr']
        self.batch_size=hparams['batch_size']
        self.opt=hparams['opt']
        self.lr_scheduler =None
        # self.train_logger = logging.getLogger(__name__)
        self.validation_recall = pl.metrics.Recall(average='macro', mdmc_average='samplewise', num_classes=4)
        self.validation_precision = pl.metrics.Precision(average='macro', mdmc_average='samplewise', num_classes=4)
        self.validation_IOU2 = pl.metrics.IoU( num_classes=4,absent_score=1,reduction='none')
        if hparams['datasetmode']== 4 or hparams['datasetmode']==8:
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
        if self.lr != 5e-4:
            print("wow!")
            print("lr=",self.lr)
        y_hat = self(x)
        y_copy = y.clone()
        # for idx,yc in enumerate(y_copy):
        #     print(y_copy.size())
        #     plt.imshow(y_copy[idx, 0, ...].cpu().numpy())
        #     plt.title(f"{idx} at training")
        #     plt.show()
        if self.modifiy_label_ON:
            for idx,z in enumerate(z_bactch):
                if z != 0:
                    # pred = torch.squeeze(y_hat)
                    pred = y_hat.permute(0,2, 3, 1)
                    pred = torch.softmax(pred[idx,...], dim=-1)
                    picked_channel = pred[idx,...].argmax(dim=-1)
                    cords = np.argwhere(picked_channel.cpu().numpy() == z)
                    realcord = []
                    #如果在原groundtruth里是背景才会修改，不是不改
                    for cord in cords:
                        if y_copy[idx,0,cord[0], cord[1]] == 0:
                            realcord.append(cord)

                    for cord in realcord:
                        y_copy[idx,0,cord[0], cord[1]] = z
                    #todo:如果是肺，右肺（label=3）再来一遍
                    if z ==2:
                        cord_zusatz=np.argwhere(picked_channel.cpu().numpy() == z+1)
                        # realcord清空
                        realcord = []
                        for cord in cord_zusatz:
                            if y_copy[idx, 0, cord[0], cord[1]] == 0:
                                realcord.append(cord)

                        for cord in realcord:
                            y_copy[idx, 0, cord[0], cord[1]] = z+1

        loss = self.loss.forward(y_hat, y_copy)
        self.log("loss", loss)
        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    def validation_step(self, batch, batch_idx, dataset_idx=0):


        x, y = batch["image"], batch["label"]
        # shape = [batch, channel, w, h]
        z_bactch= batch["leaky"]
        pred = self(x)
        y_copy = y.clone()
        # if self.modifiy_label_ON:
        #     for idx, z in enumerate(z_bactch):
        #         if z != 0:
        #             # pred = torch.squeeze(y_hat)
        #             predc = pred.permute(0, 2, 3, 1)
        #             predc = torch.softmax(predc[idx, ...], dim=-1)
        #             picked_channel = predc[idx, ...].argmax(dim=-1)
        #
        #             cords = np.argwhere(picked_channel.cpu().numpy() == z)
        #             realcord = []
        #             for cord in cords:
        #                 if y_copy[idx, 0, cord[0], cord[1]] == 0:
        #                     realcord.append(cord)
        #
        #             for cord in realcord:
        #                 y_copy[idx, 0, cord[0], cord[1]] = z
        #             # todo:如果是肺(label=2，左肺)，右肺（label=3）再来一遍
        #             if z == 2:
        #                 cord_zusatz = np.argwhere(picked_channel.cpu().numpy() == z + 1)
        #                 realcord = []
        #                 for cord in cord_zusatz:
        #                     if y_copy[idx, 0, cord[0], cord[1]] == 0:
        #                         realcord.append(cord)
        #
        #                 for cord in realcord:
        #                     y_copy[idx, 0, cord[0], cord[1]] = z + 1
        #         #

        loss = self.loss.forward(pred, y_copy)

        # argmax
        pred=torch.softmax(pred,dim=1)
        picked_channel=pred.argmax(dim=1)
        iou_individual = 0
        recall = 0
        precision = 0
        dice_individual = 0
        for index in range(picked_channel.shape[0]):
            iou_individual += self.validation_IOU2(picked_channel[index, ...], y[index, ...].long()).float()
            precision += self.validation_precision(picked_channel[index, ...], y[index, ...].long())
            dice_individual += dice_score(picked_channel[index, ...], y[index, ...].squeeze(1).long(),reduction='none',bg=True,no_fg_score=1)[:4].float()
            recall += self.validation_recall(picked_channel[index, ...], y[index, ...].long())
        iou_individual /= picked_channel.shape[0]
        precision /= picked_channel.shape[0]
        dice_individual /= picked_channel.shape[0]
        recall /= picked_channel.shape[0]
        iou_summean =torch.sum(iou_individual * self.weights.cuda())



        # self.log("loss", loss, on_step=False,on_epoch=True,prog_bar=True,logger=True)
        # self.log("recall", loss, on_step=False,on_epoch=True,prog_bar=True,logger=True)
        # self.log("precision", precision, on_step=False,on_epoch=True,prog_bar=True,logger=True)
        # self.log("iou_individual_bg", iou_individual[0], on_step=False,on_epoch=True,prog_bar=True,logger=True)
        # self.log("iou_individual_liver", iou_individual[1], on_step=False,on_epoch=True,prog_bar=True,logger=True)
        # self.log("iou_individual_left_lung", iou_individual[2], on_step=False,on_epoch=True,prog_bar=True,logger=True)
        # self.log("iou_individual_right_lung", iou_individual[3], on_step=False,on_epoch=True,prog_bar=True,logger=True)


        returndic={}
        returndic.setdefault("loss",loss)
        returndic.setdefault("recall",recall)
        returndic.setdefault("precision",precision)
        returndic.setdefault("iou_individual_bg",iou_individual[0])
        returndic.setdefault("iou_individual_liver",iou_individual[1])
        returndic.setdefault("iou_individual_left_lung",iou_individual[2])
        returndic.setdefault("iou_individual_right_lung",iou_individual[3])
        returndic.setdefault("iou_summean",iou_summean)

        returndic.setdefault("dice_individual_bg", dice_individual[0])
        returndic.setdefault("dice_individual_liver", dice_individual[1])
        returndic.setdefault("dice_individual_left_lung", dice_individual[2])
        returndic.setdefault("dice_individual_right_lung", dice_individual[3])

        return returndic

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # self.train_logger.info("Training epoch {} ends".format(self.current_epoch))
        self.log('train/loss', avg_loss)
    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recall = torch.stack([x['recall'] for x in outputs]).mean()
        avg_precision = torch.stack([x['precision'] for x in outputs]).mean()
        avg_iou_individual_bg = torch.stack([x['iou_individual_bg'] for x in outputs]).mean()
        avg_iou_individual_liver = torch.stack([x['iou_individual_liver'] for x in outputs]).mean()
        avg_iou_individual_left_lung = torch.stack([x['iou_individual_left_lung'] for x in outputs]).mean()
        avg_iou_individual_right_lung = torch.stack([x['iou_individual_right_lung'] for x in outputs]).mean()
        avg_iousummean = torch.stack([x['iou_summean'] for x in outputs]).mean()

        avg_dice_individual_bg = torch.stack([x['dice_individual_bg'] for x in outputs]).mean()
        avg_dice_individual_liver = torch.stack([x['dice_individual_liver'] for x in outputs]).mean()
        avg_dice_individual_left_lung = torch.stack([x['dice_individual_left_lung'] for x in outputs]).mean()
        avg_dice_individual_right_lung = torch.stack([x['dice_individual_right_lung'] for x in outputs]).mean()



        # self.train_logger.info(f"Validatoin epoch {self.current_epoch} ends, val_loss = {avg_loss}")
        self.log('valid/loss', avg_loss,logger=True)
        self.log('valid/recall', avg_recall,logger=True)
        self.log('valid/precision', avg_precision,logger=True)
        self.log('valid/avg_iou_individual_bg', avg_iou_individual_bg,logger=True)
        self.log('valid/avg_iou_individual_liver', avg_iou_individual_liver,logger=True)
        self.log('valid/avg_iou_individual_left_lung', avg_iou_individual_left_lung,logger=True)
        self.log('valid/avg_iou_individual_right_lung', avg_iou_individual_right_lung,logger=True)
        self.log('valid/avg_dice_individual_bg', avg_dice_individual_bg,logger=True)
        self.log('valid/avg_dice_individual_liver', avg_dice_individual_liver,logger=True)
        self.log('valid/avg_dice_individual_left_lung', avg_dice_individual_left_lung,logger=True)
        self.log('valid/avg_dice_individual_right_lung', avg_dice_individual_right_lung,logger=True)
        self.log('avg_iousummean', avg_iousummean,logger=True)

    def configure_optimizers(self):
        if self.opt=='Adam':
            print(f'[INFO] Adam will be used ,lr = {self.lr}')
            # return torch.optim.Adam(self.parameters(), lr=self.lr)
            optimizer=torch.optim.Adam(self.parameters(), lr=self.lr)
            return optimizer

            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2,mode='max', factor=0.9, verbose=True)
            # scheduler = {
            #     'scheduler': lr_scheduler,
            #     'reduce_on_plateau': True,
            #     # val_checkpoint_on is val_loss passed in as checkpoint_on
            #     'monitor': 'avg_iousummean'
            # }
            #
            # return [optimizer], [scheduler]
        else:
            print(f'[INFO] SGD will be used ,lr = {self.lr}')
            optimizer= torch.optim.SGD(self.parameters(), lr=self.lr,momentum=0.9,)

            return optimizer
            # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='max', factor=0.9,
            #                                                           verbose=True)
            # scheduler = {
            #     'scheduler': lr_scheduler,
            #     'reduce_on_plateau': True,
            #     # val_checkpoint_on is val_loss passed in as checkpoint_on
            #     'monitor': 'avg_iousummean'
            # }
            #
            # return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx, dataset_idx=None):
        x, y = batch['image'], batch['label']
        y_hat = self(x)

        pred=torch.softmax(y_hat,dim=1)
        picked_channel=pred.argmax(dim=1)
        iou_individual=0
        for index in range(picked_channel.shape[0]):
            iou_individual += self.validation_IOU2(picked_channel[index,...], y[index,...].long())
        iou_individual = iou_individual/picked_channel.shape[0]
        iou_summean =torch.sum(iou_individual * self.weights.cuda())

        # dice=dice_score(torch.softmax(y_hat, dim=1),y.squeeze(1).long(),reduction='none',bg=True,no_fg_score=1)[:4]
        show=0
        if show:
            for index in range(x.shape[0]):
                fig, axs = plt.subplots(1,4)

                axs[0].imshow(picked_channel[index,...].cpu() *0.5+x[index,0,...].cpu() *0.5)
                axs[1].imshow(picked_channel[index,...].cpu() )
                axs[2].imshow(x[index,0,...].cpu() )
                axs[3].imshow(y[index,0,...].cpu() )
                axs[0].set_title('blend')
                axs[1].set_title('result')
                axs[2].set_title('img')
                axs[3].set_title('ground truth')
                plt.show()

        # loss = self.loss.forward(y_hat, y)
        # self.log("iou_individual_bg", iou_individual[0], on_step=False, logger=True)
        # self.log("iou_individual_liver", iou_individual[1], on_step=False,  logger=True)
        # self.log("iou_individual_left_lung", iou_individual[2], on_step=False,logger=True)
        # self.log("iou_individual_right_lung", iou_individual[3], on_step=False, logger=True)

        returndic={}

        returndic.setdefault("iou_individual_bg",iou_individual[0])
        returndic.setdefault("iou_individual_liver",iou_individual[1])
        returndic.setdefault("iou_individual_left_lung",iou_individual[2])
        returndic.setdefault("iou_individual_right_lung",iou_individual[3])
        returndic.setdefault("iou_summean",iou_summean)


        return returndic

    def test_epoch_end(self, outputs):
        avg_iou_individual_bg = torch.stack([x['iou_individual_bg'] for x in outputs]).mean()
        avg_iou_individual_liver = torch.stack([x['iou_individual_liver'] for x in outputs]).mean()
        avg_iou_individual_left_lung = torch.stack([x['iou_individual_left_lung'] for x in outputs]).mean()
        avg_iou_individual_right_lung = torch.stack([x['iou_individual_right_lung'] for x in outputs]).mean()
        avg_iousummean = torch.stack([x['iou_summean'] for x in outputs]).mean()
        self.log('avg_iousummean', avg_iousummean,logger=True)
        self.log('valid/avg_iou_individual_bg', avg_iou_individual_bg, logger=True)
        self.log('valid/avg_iou_individual_liver', avg_iou_individual_liver, logger=True)
        self.log('valid/avg_iou_individual_left_lung', avg_iou_individual_left_lung, logger=True)
        self.log('valid/avg_iou_individual_right_lung', avg_iou_individual_right_lung, logger=True)

        # self.train_logger.info("test epoch {} ends,iou={},{},{},{}".format(self.current_epoch, avg_iou_individual_bg,
        #                                                                    avg_iou_individual_liver,
        #                                                                    avg_iou_individual_left_lung,
        #                                                                    avg_iou_individual_right_lung))





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