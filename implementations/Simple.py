from implementations.Implementation import Implementation

class Simple(Implementation):
    """Standard implementation that simply averages pixels."""

    def __init__(self):
        """Intialize class."""
        pass


    def run(self, imgdata, size):
        self.imgdata = imgdata
        self.size = size

        i = 0
        for x in range(0, len(self.imgdata), size[0]):
            for y in range(x + i, x + size[0], 2):
                self.imgdata[y] = self.fillpixel(y)
            if i == 0:
                i = 1
            else:
                i = 0

        return self.imgdata

    def fillpixel(self, index):

        if index >= self.size[0]:
            above = index - self.size[0]
        else:
            above = None

        if index <= self.size[0] * (self.size[1] - 1):
            below = index + self.size[0]
        else:
            below = None

        if index >= 1:
            left = index - 1
        else:
            left = None

        if index < self.size[0]:
            right = index + 1
        else:
            right = None

        items = [above,below,left,right]
        r,g,b,count = 0,0,0,0
        for item in items:
            if item is not None:
                r += self.imgdata[item][0]
                g += self.imgdata[item][1]
                b += self.imgdata[item][2]
                count += 1

        return (int(r/count),int(g/count),int(b/count))

