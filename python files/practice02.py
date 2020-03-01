# <20.2.28> by KH

"""
21 page
keras feature engineering
"""
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rotation_range = 45,           # 회전 (0~180)
                             width_shift_range=0.25,        #
                             height_shift_range = 0.25,
                             rescale = 1./255,
                             shear_range = 0.3,             # shearing 해줌 -> 기울어짐
                             zoom_range = 0.3,
                             horizontal_flip = True,
                             fill_mode = 'nearest')
