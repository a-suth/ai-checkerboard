import numpy as np
from PIL import Image
from implementations.Simple import Simple
from utilities.Checkerboard import CheckerBoard
from implementations.TensorScale import TensorScale


checkerboarder = CheckerBoard()

def basic():
    with Image.open('test3.png') as img:
        
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
    #tensor = TensorScale()
    #tensor.run()

    with Image.open('Capture.PNG') as img:

        img_data = np.asarray(img)   
        img_data.flags.writeable = False     
        
        checked_data = checkerboarder.run(img_data)
        
        simple = Simple()
        #tensor_result = tensor.fill(checked_data)

        print(checkerboarder.get_difference(img_data, checked_data))

        im2 = Image.new(img.mode, img.size)
        im2 = Image.fromarray(np.uint8(checked_data))
        im2.show()
        im2.save('new.png')
        

generate_data_csv()
#basic()

