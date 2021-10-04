from tensorboard.backend.event_processing import event_accumulator
import argparse,os
import pandas as pd
from matplotlib import pyplot
import numpy as np
from tqdm import tqdm
def get_list(data):
    data_list=[]
    for i in data:
        data_list.append(i.value)
    return data_list

def main():
    # load log data
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in-path', type=str, default=r'F:\Forschung\multiorganseg\tensorboard', help='Tensorboard event files or a single tensorboard '
                                                                   'file location')
    parser.add_argument('--ex-path', type=str, default=r'F:\Forschung\multiorganseg\others\result.csv', help='location to save the exported data')

    args = parser.parse_args()
    path_list= os.listdir(args.in_path)
    mode1,mode2,mode3,mode4,mode5,mode6=[],[],[],[],[],[]
    modeList=[mode1,mode2,mode3,mode4,mode5,mode6]
    a=[]
    for path,data in zip (path_list,modeList):
        event_data = event_accumulator.EventAccumulator(os.path.join(args.in_path,path))  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        # keys = event_data.scalars.Keys()  # get all tags,save in a list
        temp=event_data.scalars.Items('valid/loss')
        data=get_list(temp)
        a.append(data)
        # print(data[:20])
    fig = pyplot.figure()
    # pyplot.subplot(121)
    pyplot.title("valid/loss", fontsize=15)  # 设置子图标题
    color_map=['b', 'g', 'r', 'c', 'm','y']
    label_map=['mode1', 'mode2', 'mode3', 'mode4', 'mode5','mode6']

    for i,c,l in zip (a,color_map,label_map):
        print(len(i))
        idx=np.arange(0,len(i))
        pyplot.plot(idx, i, c, label=l)
    pyplot.legend()
    pyplot.grid()
    pyplot.show()
    # df['loss'] = pd.DataFrame(i.value)
    # for key in tqdm(keys):
    #     # print(key)
    #     # if key == 'train/total_loss_iter':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
    #     df[key] = pd.DataFrame(event_data.Scalars(key)).value

    # df.to_csv(args.ex_path)

    print("Tensorboard data exported successfully")
if __name__ == '__main__':
    main()