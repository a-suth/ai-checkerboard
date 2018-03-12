import numpy as np
from PIL import Image
from implementations.Simple import Simple
from utilities.Checkerboard import CheckerBoard
from implementations.TensorScale import TensorScale


checkerboarder = CheckerBoard()

def basic():
    with Image.open('TEST.png') as img:
        
        img_data = np.asarray(img)   
        img_data.flags.writeable = False     
        
        checked_data = checkerboarder.run(img_data)
        
        simple = Simple()
        simple_result = simple.run(checked_data)

        print(checkerboarder.get_difference(img_data, simple_result))

        im2 = Image.new(img.mode, img.size)
        im2 = Image.fromarray(np.uint8(simple_result))
        im2.show()
        im2.save('new.png')
        
        


def generate_data_csv():
    tensor = TensorScale()
    tensor.generate_data_csv()

#generate_data_csv()
basic()

