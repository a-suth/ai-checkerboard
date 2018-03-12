from implementations.Implementation import Implementation
import numpy as np

class Simple(Implementation):
    """Standard implementation that simply averages pixels."""

    def __init__(self):
        """Intialize class."""
        pass


    def run(self, imgdata):
        self.imgdata = imgdata.copy()

        i = 0
        for x in range(0, len(self.imgdata)):       # for each row
            row = self.imgdata[x]
            print('\r\r%s%% done' % round((x/len(self.imgdata)) * 100, 0), end='')
            for y in range(i, len(row), 2):
                #row[y] = self.fillpixel(x,y)
                self.fillpixel(x,y)
            if i == 0:
                i = 1
            else:
                i = 0
        return self.imgdata

    def fillpixel(self, row, col):
        above,below,left,right = self.get_surrounding(row, col)
        r,g,b,count = 0,0,0,0
        
        for item in [above,below,left,right]:
            if item is not None:
                pixel = self.imgdata[item[0],item[1]]
                r += pixel[0]
                g += pixel[1]
                b += pixel[2]
                count += 1

        self.imgdata[row,col] =  (int(r/count),int(g/count),int(b/count),0)



