import os
import sys
import pickle
import matplotlib.pyplot as plt
import torch
import torchmetrics
from measures import calculate_eval_matrix, calculate_dice, calculate_IoU
import torchvision
import numpy as np
import pytorch_lightning as pl
from torchmetrics.functional import dice_score
from skimage import color

torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.append(os.path.dirname(__file__))


class BasetRAIN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.model = None
        self.loss = None
        self.lossflag = hparams['loss']
        self.hparams.update(hparams)
        self.weights = torch.tensor([0.1, 1.0, 1.0, 1.0])
        self.lr = hparams['lr']
        self.batch_size = hparams['batch_size']
        self.opt = hparams['opt']
        self.lungrecord = np.empty((1, 0))
        self.datamode = hparams['datasetmode']
        # self.train_logger = logging.getLogger(__name__)
        self.validation_recall = torchmetrics.Recall(average='macro', mdmc_average='samplewise', num_classes=4)
        self.validation_precision = torchmetrics.Precision(average='macro', mdmc_average='samplewise', num_classes=4)
        self.validation_Accuracy = torchmetrics.Accuracy(num_classes=4)
        self.validation_IOU2 = torchmetrics.IoU(num_classes=4, absent_score=1)
        self.validation_IOU = torchmetrics.IoU(num_classes=4, absent_score=1,reduction='none')
        if hparams['datasetmode'] == 4 or hparams['datasetmode'] == 8 or  hparams['datasetmode'] ==6:
            self.modifiy_label_ON = True
            print(f'[INFO] modifiy_label_ON={self.modifiy_label_ON}')
        else:
            self.modifiy_label_ON = False
            print(f'[INFO] modifiy_label_ON={self.modifiy_label_ON}')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, dataset_idx=None):
        x, y = batch["image"], batch["label"]
        z_bactch = batch["leaky"]
        # if self.modifiy_label_ON:
        #     lr =self.lr
        #     if self.current_epoch % 8 ==0 and self.current_epoch // 8 > 0:
        #         self.lr *=0.8
        #     if self.lr != lr:
        #         print("new lr=",self.lr)
        y_hat = self(x)

        y_copy = y.clone()
        predlist = []

        if self.modifiy_label_ON:
            for idx, z in enumerate(z_bactch):
                z = float(z)
                if z != 0:
                    picked_channel = y_hat[idx, ...].argmax(dim=0)
                    cord_not_sure =picked_channel == z
                    assert len(picked_channel.size()) == 2
                    '''
                   --------------------x 
                    |
                    |
                    |
                    y
                    '''
                    cord_zero_InTarget = y[idx,0,...]== 0

                    # 如果在原groundtruth里是背景才会修改，不是不改

                    realcord=torch.bitwise_and(cord_not_sure,cord_zero_InTarget)
                    temp= realcord==True
                    # if torch.max(y_copy[idx,...])!=0:
                    y_copy[idx, 0][realcord==True] = z
                    print(y_copy.size())

                    # todo:如果是肺，右肺（label=3）再来一遍
                    if z == 2:
                        cord_not_sure = picked_channel == 3
                        cord_zero_InTarget = y[idx,0,...] == 0
                        realcord = torch.bitwise_and(cord_not_sure, cord_zero_InTarget)==True
                        y_copy[idx, 0][realcord==True] = 3

                else:
                    if self.datamode == 4:
                        print('mode4')
                        continue
                    else:
                        picked_channel = None
                        print('identity mark z is ：', z)
                        raise ValueError('Data error')

                # predlist.append(picked_channel)
            # if self.current_epoch>3:
            #     plt.figure()
            #     if z_bactch[0] == 2:
            #         text='Lung'
            #     else:
            #          text='Liver'
            #     plt.imshow(x[0, 0, ...].cpu().numpy(), cmap='Blues')
            #     plt.title(f'Input data Missing label={text}')
            #     plt.show()
            #
            #     plt.imshow(predlist[0].cpu().numpy(), cmap='Blues')
            #     class_pred=torch.unique(predlist[0])
            #     plt.title(f'Prediction,has{class_pred}')
            #     plt.show()
            #
            #     plt.imshow(y[0, 0, ...].cpu().numpy(), cmap='Blues')
            #     plt.title(f'Original Ground Truth')
            #     plt.show()
            #
            #     plt.imshow(y_copy[0, 0, ...].cpu().numpy(), cmap='Blues')
            #     class_pred2=torch.unique(y_copy[0, 0, ...])
            #
            #     plt.title(f'Simulate non-fully annotated dataset,has{class_pred2}')
            #     plt.show()
            #
            #     fig, axs = plt.subplots(1, 4)
            #     for i in range(1):
            #         axs[0].imshow(predlist[i].cpu().numpy(), cmap='Blues')
            #         if z_bactch[i]==2:
            #             text='Lung'
            #         else:
            #             text='Liver'
            #         axs[0].set_title(f'Missing label={text}')
            #         axs[1].imshow(y[i, 0, ...].cpu().numpy(), cmap='Blues')
            #         axs[1].set_title(f'Original Ground Truth')
            #         axs[2].imshow(y_copy[i, 0, ...].cpu().numpy(), cmap='Blues')
            #         axs[2].set_title(f'New Ground Truth')
            #         axs[3].imshow(x[i, 0, ...].cpu().numpy(), cmap='Blues')
            #         axs[3].set_title(f'Input data')
            #     plt.show()
        if self.lossflag == 'Dice':
            y_hat = torch.sigmoid(y_hat)
        loss = self.loss.forward(y_hat, y_copy)
        self.log("loss", loss)
        return {"loss": loss}

    # def validation_step(self, batch, batch_idx):
    def validation_step(self, batch, batch_idx, dataset_idx=0):

        x, y = batch["image"], batch["label"]
        # shape = [batch, channel, w, h]
        # z_bactch= batch["leaky"]

        pred = self(x)

        # argmax
        pred = torch.softmax(pred, dim=1)
        picked_channel = pred.argmax(dim=1)

        # for index in range(picked_channel.shape[0]):
        #     iou_individual += self.validation_IOU2(picked_channel[index, ...], y[index, ...].long()).float()
        #     precision += self.validation_precision(picked_channel[index, ...], y[index, ...].long())
        #     dice_individual += dice_score(picked_channel[index, ...], y[index, ...].squeeze(1).long(), reduction='none',
        #                                   bg=True, no_fg_score=1)[:4].float()
        #     recall += self.validation_recall(picked_channel[index, ...], y[index, ...].long())
        precision =self.validation_precision(picked_channel, y.long())
        dice_summean =dice_score(picked_channel, y.squeeze(1).long(),
                                           bg=True, no_fg_score=1).float()
        recall =self.validation_recall(picked_channel, y.long())
        iou_summean = self.validation_IOU2(picked_channel, y.long())

        if self.lossflag == 'Dice':
            pred = torch.sigmoid(self(x))
        loss = self.loss.forward(pred, y)

        returndic = {}
        returndic.setdefault("loss", loss)
        returndic.setdefault("recall", recall)
        returndic.setdefault("precision", precision)
        # returndic.setdefault("iou_individual_bg", iou_individual[0])
        # returndic.setdefault("iou_individual_liver", iou_individual[1])
        # returndic.setdefault("iou_individual_left_lung", iou_individual[2])
        # returndic.setdefault("iou_individual_right_lung", iou_individual[3])
        returndic.setdefault("iou_summean", iou_summean)
        returndic.setdefault("dice_summean", dice_summean)
        #
        # returndic.setdefault("dice_individual_bg", dice_individual[0])
        # returndic.setdefault("dice_individual_liver", dice_individual[1])
        # returndic.setdefault("dice_individual_left_lung", dice_individual[2])
        # returndic.setdefault("dice_individual_right_lung", dice_individual[3])

        return returndic

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # self.train_logger.info("Training epoch {} ends".format(self.current_epoch))
        self.log('train/loss', avg_loss)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_recall = torch.stack([x['recall'] for x in outputs]).mean()
        avg_precision = torch.stack([x['precision'] for x in outputs]).mean()
        # avg_iou_individual_bg = torch.stack([x['iou_individual_bg'] for x in outputs]).mean()
        # avg_iou_individual_liver = torch.stack([x['iou_individual_liver'] for x in outputs]).mean()
        # avg_iou_individual_left_lung = torch.stack([x['iou_individual_left_lung'] for x in outputs]).mean()
        # avg_iou_individual_right_lung = torch.stack([x['iou_individual_right_lung'] for x in outputs]).mean()
        avg_iousummean = torch.stack([x['iou_summean'] for x in outputs]).mean()
        avg_dice_summean = torch.stack([x['dice_summean'] for x in outputs]).mean()

        # avg_dice_individual_bg = torch.stack([x['dice_individual_bg'] for x in outputs]).mean()
        # avg_dice_individual_liver = torch.stack([x['dice_individual_liver'] for x in outputs]).mean()
        # avg_dice_individual_left_lung = torch.stack([x['dice_individual_left_lung'] for x in outputs]).mean()
        # avg_dice_individual_right_lung = torch.stack([x['dice_individual_right_lung'] for x in outputs]).mean()

        # self.train_logger.info(f"Validatoin epoch {self.current_epoch} ends, val_loss = {avg_loss}")
        self.log('valid/loss', avg_loss, logger=True)
        self.log('valid/recall', avg_recall, logger=True)
        self.log('valid/precision', avg_precision, logger=True)
        # self.log('valid/avg_iou_individual_bg', avg_iou_individual_bg, logger=True)
        # self.log('valid/avg_iou_individual_liver', avg_iou_individual_liver, logger=True)
        # self.log('valid/avg_iou_individual_left_lung', avg_iou_individual_left_lung, logger=True)
        # self.log('valid/avg_iou_individual_right_lung', avg_iou_individual_right_lung, logger=True)
        # self.log('valid/avg_dice_individual_bg', avg_dice_individual_bg, logger=True)
        # self.log('valid/avg_dice_individual_liver', avg_dice_individual_liver, logger=True)
        # self.log('valid/avg_dice_individual_left_lung', avg_dice_individual_left_lung, logger=True)
        # self.log('valid/avg_dice_individual_right_lung', avg_dice_individual_right_lung, logger=True)
        self.log('avg_iousummean', avg_iousummean, logger=True)
        self.log('avg_iousummean', avg_dice_summean, logger=True)

    def configure_optimizers(self):
        if self.opt == 'Adam':
            print(f'[INFO] Adam will be used ,lr = {self.lr}')
            # return torch.optim.Adam(self.parameters(), lr=self.lr)
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, )

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

    def show(self, x, picked_channel, y, index):
        fig, axs = plt.subplots(1, 4)

        axs[0].imshow(picked_channel[index, ...].cpu() * 0.5 + x[index, 0, ...].cpu() * 0.5)
        axs[1].imshow(picked_channel[index, ...].cpu())
        axs[2].imshow(x[index, 0, ...].cpu())
        axs[3].imshow(y[index, 0, ...].cpu())
        axs[0].set_title('blend')
        axs[1].set_title('result')
        axs[2].set_title('img')
        axs[3].set_title('ground truth')
        plt.show()

    def test_step(self, batch, batch_idx, dataset_idx=None):
        '''
        经测试，用for循环每张图做acc然后手动去平均和直接扔进去效果一样
        但其他参数不是这样
        saveimg (batch,colorchannel,h,w)
        '''

        def mapping_color(img):
            '''
            自己写的，速度快不少，但要自己规定colormap，也可以把制定colormap拿出来单独用randint做，
            但是不能保证一个series里每次运行生成的colormap都一样，或许可以用种子点？
            反正类少还是可以考虑用这个
                    '''
            color_map = [[255, 0, 0], [255, 255, 0], [0, 0, 255]]
            for label in [1, 2, 3]:
                cord_1 = torch.where(img[:, 0, ...] == label)
                img[:, 0, cord_1[1], cord_1[2]] = color_map[label - 1][0]
                img[:, 1, cord_1[1], cord_1[2]] = color_map[label - 1][1]
                img[:, 2, cord_1[1], cord_1[2]] = color_map[label - 1][2]
            return img

        def label2rgb(img):
            '''
            用了skimage速度慢点，但是不用操心有几个类

            '''
            templist = []
            for i in range(img.size()[0]):
                temp = color.label2rgb(img[i].numpy(), bg_label=0)
                templist.append(torch.tensor(temp))
            result = torch.stack((templist), dim=0)
            result = torch.moveaxis(result, 3, 1)
            return result

        x, y = batch['image'], batch['label']
        y = y.squeeze(1)
        pred = torch.sigmoid(self(x))

        picked_channel = pred.argmax(dim=1)
        iou_individual = 0
        recall = 0
        precision = 0
        dice_individual = 0
        acc = 0
        for index in range(picked_channel.shape[0]):
            iou_individual += self.validation_IOU(picked_channel[index, ...], y[index, ...].int()).float()
            precision += self.validation_precision(picked_channel[index, ...], y[index, ...].int())
            acc += self.validation_Accuracy(picked_channel[index, ...], y[index, ...].int())

            dice_individual += dice_score(picked_channel[index, ...], y[index, ...].squeeze(1).int(), reduction='none',
                                          bg=True, no_fg_score=1)[:4].float()
            recall += self.validation_recall(picked_channel[index, ...], y[index, ...].int())
        iou_individual /= picked_channel.shape[0]
        precision /= picked_channel.shape[0]
        acc /= picked_channel.shape[0]
        dice_individual /= picked_channel.shape[0]
        recall /= picked_channel.shape[0]
        iou_summean = torch.sum(iou_individual * self.weights.cuda())

        result_saved = torch.cat((picked_channel, y), dim=1)
        # result_saved=label2rgb(result_saved.cpu())

        result_saved = torch.unsqueeze(result_saved, dim=1)

        result_saved = torch.hstack(
            (result_saved, result_saved, result_saved))
        result_saved = mapping_color(result_saved).cpu().float()
        folder = "saved_images/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        torchvision.utils.save_image(
            result_saved, f"{folder}/infer_result{batch_idx}.jpg"
        )

        returndic = {}
        returndic.setdefault("recall", recall)
        returndic.setdefault("precision", precision)
        returndic.setdefault("acc", acc)

        returndic.setdefault("picked_channel", picked_channel)
        returndic.setdefault("y", y.int())
        #
        returndic.setdefault("iou_individual_bg", iou_individual[0])
        returndic.setdefault("iou_individual_liver", iou_individual[1])
        returndic.setdefault("iou_individual_left_lung", iou_individual[2])
        returndic.setdefault("iou_individual_right_lung", iou_individual[3])
        returndic.setdefault("iou_summean", iou_summean)

        return returndic

    def test_epoch_end(self, outputs):
        '''
        outputs 是所有step输出的集合，tuple类型，长度为step数，神了，
        我也不知道pl咋做的，反正就是这个epoch里的step的集合。。我还以为要用on step end 才行，结果这个api就行、、

        '''
        print('test epoch end')

        outpick = torch.cat([x['picked_channel'] for x in outputs], dim=0)
        outy = torch.cat([x['y'] for x in outputs], dim=0)
        mat = calculate_eval_matrix(num_cls=4, labels=outy.cpu().numpy(), predictions=outpick.cpu().numpy())
        iou = calculate_IoU(mat)
        dice = calculate_dice(mat)

        avg_iou_individual_bg = torch.stack([x['iou_individual_bg'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        avg_iou_individual_liver = torch.stack([x['iou_individual_liver'] for x in outputs]).mean()
        avg_iou_individual_left_lung = torch.stack([x['iou_individual_left_lung'] for x in outputs]).mean()
        avg_iou_individual_right_lung = torch.stack([x['iou_individual_right_lung'] for x in outputs]).mean()
        avg_iousummean = torch.stack([x['iou_summean'] for x in outputs]).mean()
        self.log('avg_iousummean', avg_iousummean, logger=True)
        self.log('acc', avg_acc, logger=True)
        self.log('valid/avg_iou_individual_bg', avg_iou_individual_bg, logger=True)
        self.log('valid/avg_iou_individual_liver', avg_iou_individual_liver, logger=True)
        self.log('valid/avg_iou_individual_left_lung', avg_iou_individual_left_lung, logger=True)
        self.log('valid/avg_iou_individual_right_lung', avg_iou_individual_right_lung, logger=True)
        with open("saved_images/log.txt", "w") as t:
            t.writelines('iou:' + str(iou) + '\n')
            t.writelines('dice:' + str(dice) + '\n')
            t.writelines('avg_iousummean:' + str(np.sum(iou)))
        print(iou)
        print(dice)

    def on_test_end(self) -> None:
        if os.path.exists("lungrecord.pkl"):
            print('del..lungrecord.pkl')
            os.remove("lungrecord.pkl")
        with open("lungrecord.pkl", 'wb') as f:
            pickle.dump(self.lungrecord, f)
        # print(self.lungrecord.shape)
        print('test end')


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
