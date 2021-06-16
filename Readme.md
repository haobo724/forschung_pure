ds.py负责数据集预处理，包括数据的各种变换，把数据集根据病人代号分组0-34，在后面的数据集调用时通过病人的index从而调整数据集的组成的多少

custom_transform.py 是继承于monai.transforms.compose下的transform，里面定义了两个比较重要的transform，一个是标记当前数据集/病人是否是完整数据集，即无缺失label。第二个transform是将如果标记为缺省数据集的groundtruth修改成符合要求的结果，如肝设置缺省，那原groundtruth图像中所有等于1的像素点置为0（背景）

song.dataset.py 负责返回需要的dataloader，继承于pl.LightningDataModule,初始化参数里的mode变量决定返回的dataloader内数据组成

helpers.py 没用

----------------------------------------------
根目录下：

base_train2D.py 依赖pl框架的训练unit，可通过传参对训练过程进行调整，现可以调整：
optmizer，loss, lr。 目前内置的metric 有iou,recall,dice,precision.

train步骤核心：如果是mode4或8则开启优化算法，将predicate结果与groundtruth相结合

val步骤的loss计算则一直使用未经任何像素值更改的原groundtruth图像，且在每两个train_epoch结束后才进行一次val epoch计算metric并保留top2模型（basetrain_song.py内定义）

basetrain_song.py：具体实例化和细化关于训练的相关的参数，包括log，trainer和传参设置

infer_song.py: 对传入的模型inference，默认数据集mode为5，十个病人组成
在teststep里可以通过调整show变量的选择展示或不展示结果

loss.py: 定义diceloss 相对于原版修复了一个不适配的bug（row:48)

其他没提过的py都没用到，用来临时测试python一些语法的

