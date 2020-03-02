# <20.3.1> by KH

'''
80 page
Keras for expression recognition
파일이 없어서 안 돌려봄.
'''

import os
import pandas as pd
from PIL import Image

# Pixel values range from 0 to 255 ( 0 is normally black and 255 is white)
os.system("mkdir data")
os.system("cd data")
os.system("mkdir raw")
basedir = os.path.join('data','raw')
file_origin = os.path.join(basedir, 'fer2013.csv')
data_raw = pd.read_csv(file_origin)

data_input = pd.DataFrame(data_raw, columns = ['emotions', 'pixels', 'Usage'])

data_input.rename({'Usage':'usage'}, inplace = True)
data_input.head()

label_map = {
    0: '0_Anger',
    1: '1_Disgust',
    2: '2_Fear',\
    3: '3_Happy',
    4: '4_Neutral',
    5: '5_Sad',\
    6: '6_Surprise'
}

# Creating the folders
output_folers = data_input['Usage'].unique().tolist()
all_folders = []

for folder in output_folders:
    for label in label_map:
        all_folders.append(os.path.join(basedir, folder, label_map[label]))

for folder in all_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

    else:
        print('Folder {} exists already'.format(folder))

counter_error = 0
counter_correct = 0

def save_image(np_array_flat, file_name):
    try:
        im = Image.fromarray(np_array_flat)
        im.save(file_name)

    except AttributeError as e:
        print(e)
        return

for folder in all_folders:
    emotion = foler.split('/')[-1]
    usage = folder.split('/')[-2]

    for key, value in label_map.items():
        if value == emotion:
            emotion_id = key

    df_to_save = data_input.reset_index()[data_input.Usage == usage][data_input.emotion == emotion_id]
    print('saving in: ', folder, ' size: ', df_to_save.shape)
    df_to_save['image'] = df_to_save.pixels.apply(to_image)
    df_to_save['file_name'] = folder + '/image_' + df_to_save.index.map(str) + '_' + df_to_save.emotion.apply(str) + '-' + df_to_save.emotion.apply(lambda x: label_map[x]) + '.png'
    df_to_save[['image', 'file_name']].apply(lambda x: save_image(x.image, x.file_name), axis=1)
    df_to_save.apply(lambda x: save_image(x.pixels, os.path.join(basedir,x.file_name)), axis=1)
