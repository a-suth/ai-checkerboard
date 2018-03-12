import numpy as np
from PIL import Image
from implementations.Simple import Simple
from implementations.Checkerboard import CheckerBoard

with Image.open('1500052540076.png') as img:
    img_data = list(img.getdata())
    
    checkerboarder = CheckerBoard()
    new_data = checkerboarder.run(img_data, img.size)
    
    simple = Simple()
    simple_result = simple.run(img_data, img.size)


    im2 = Image.new(img.mode, img.size)
    im2.putdata(new_data)
    im2.show()
    im2.save('new.png')
    