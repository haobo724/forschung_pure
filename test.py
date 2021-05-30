list1=[0,0,1,3,4]
dataset_mode=1
fulllabeled_name_sub = list1[:1] if dataset_mode == 1 else list1[:2]
print(fulllabeled_name_sub)
