import os
import tensorflow as ts
from PIL import Image
from implementations.Implementation import Implementation

import numpy as np

class TensorScale(Implementation):
    def __init__(self):
        self.data = []

    def generate_data_csv(self):
        print('loading files...')
        for filename in os.listdir('./trainingimgs'):
            with Image.open(filename) as img:
                img_data = np.asarray(img)   
                img_data.flags.writeable = False

                for x in range(0, len(self.imgdata)):       # for each row
                    row = self.imgdata[x]
                    for y in range(i, len(row), 1):
                        above,below,left,right = self.get_surrounding(x,y)
                        if None in [above,below,left,right]:
                            pass
                        else:
                            print(above,below,left,right,x)
                            data = numpy.array([img_data[above][:3],img_data[below][:3],img_data[left][:3],img_data[right][:3],img_data[x][:3]])
                            print(data)

            