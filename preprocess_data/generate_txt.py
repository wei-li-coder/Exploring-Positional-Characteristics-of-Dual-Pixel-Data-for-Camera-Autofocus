import os
import random

# generate patch txt
train_dir = '/data/wl/autofocus/learn2focus/dataset/train_set/'
dirs_1 = os.listdir(train_dir)
dirs_1.sort()
# dirs_1 = ['apt1_0',...] scene name
train_txt = open('/data/wl/autofocus/learn2focus/dataset/train_set.txt','a')
for dir_1 in dirs_1:
    dirs_2 = os.listdir(train_dir + dir_1 + '/single')
    dirs_2.sort(key=int)
    # dirs_2 = ['0', '1', ..., '48']
    for dir_2 in dirs_2:
        files = os.listdir(train_dir + dir_1 + '/single/' + dir_2)
        files.sort()
        for file in files:
            name =  train_dir +  dir_1 + '/single/' + dir_2 + '/' + file + ' ' + dir_2 +'\n'
            train_txt.write(name)

train_txt.close()

test_dir = '/data/wl/autofocus/learn2focus/dataset/test_set/'
dirs_1 = os.listdir(test_dir)
dirs_1.sort()
# dirs_1 = ['0', '1', ..., '48']
test_txt = open('/data/wl/autofocus/learn2focus/dataset/test_set.txt','a')
for dir_1 in dirs_1:
    dirs_2 = os.listdir(test_dir + dir_1 + '/single')
    dirs_2.sort(key=int)
    for dir_2 in dirs_2:
        files = os.listdir(test_dir + dir_1 + '/single/' + dir_2)
        files.sort()
        for file in files:
            name =  test_dir +  dir_1 + '/single/' + dir_2 + '/' + file + ' ' + dir_2 +'\n'
            test_txt.write(name)

test_txt.close()
