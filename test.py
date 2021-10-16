from tensorboard.backend.event_processing import event_accumulator
import argparse, os
import pandas as pd
from matplotlib import pyplot
import numpy as np
from tqdm import tqdm


def get_list(data):
    data_list = []
    for i in data:
        data_list.append(i.value)
    return data_list


def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-path', type=str, default=r'F:\Forschung\multiorganseg\tensorboard',
                        help='Tensorboard event files or a single tensorboard '
                             'file location')
    parser.add_argument('--ex-path', type=str, default=r'F:\Forschung\multiorganseg\others\result.csv',
                        help='location to save the exported data')

    args = parser.parse_args()
    path_list = os.listdir(args.in_path)
    mode1, mode2, mode3, mode4, mode5, mode6 = [], [], [], [], [], []
    modeList = [mode1, mode2, mode3, mode4, mode5, mode6]
    a = []
    for path, data in zip(path_list, modeList):
        event_data = event_accumulator.EventAccumulator(
            os.path.join(args.in_path, path))  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        # keys = event_data.scalars.Keys()  # get all tags,save in a list
        temp = event_data.scalars.Items('valid/loss')
        data = get_list(temp)
        a.append(data)
        # print(data[:20])
    fig = pyplot.figure()
    # pyplot.subplot(121)
    pyplot.title("valid/loss", fontsize=15)  # 设置子图标题
    color_map = ['b', 'g', 'r', 'c', 'm', 'y']
    label_map = ['mode1', 'mode2', 'mode3', 'mode4', 'mode5', 'mode6']

    for i, c, l in zip(a, color_map, label_map):
        print(len(i))
        idx = np.arange(0, len(i))
        pyplot.plot(idx, i, c, label=l)
    pyplot.legend()
    pyplot.grid()
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')

    pyplot.show()

    print("Tensorboard data exported successfully")


def plot_bar():
    mode1, mode3, mode4, mode5, mode6 = [0.996, 0.934, 0.964, 0.936], \
                                        [0.987, 0.893, 0.849, 0.847], \
                                        [0.995, 0.919, 0.938, 0.911], \
                                        [0.986, 0.850, 0.833, 0.835], \
                                        [0.988, 0.889, 0.919, 0.899]
    modelist=[mode1, mode3, mode4, mode5, mode6]
    width = 0.15
    index = np.arange(len(mode1))
    # for mode in modelist:
    #     mode[:]=mode[:]-np.min(mode)
    r1 = pyplot.bar(index, mode1, width, color='#F27970', label='mode1')
    r2 = pyplot.bar(index + width, mode3, width, color='#BB9727', label='mode3')
    r3 = pyplot.bar(index + width + width, mode4, width, color='#8983bf', label='mode4')
    r4 = pyplot.bar(index + width + width+width, mode5, width, color='#32B897', label='mode5')
    r5 = pyplot.bar(index + width + width+width+width, mode6, width, color='#c76da2', label='mode6')
    pyplot.xticks(index+0.3,['bg','liver','left lung','right lung'],fontsize=25)
    pyplot.yticks(fontsize=25)
    pyplot.ylim(0,1.1)
    pyplot.xlabel('label class')
    pyplot.ylabel('IoU')
    pyplot.title('IoU comparison of four classes in five different modes ',fontsize=30)
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    plot_bar()
