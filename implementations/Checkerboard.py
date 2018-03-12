class CheckerBoard:

    def __init__(self):
        pass

    def run(self, imgdata, size):
        self.imgdata = imgdata

        i = 0
        for x in range(0, len(self.imgdata), size[0]):
            for y in range(x + i, x + size[0], 2):
                self.imgdata[y] = (255,0,0)
            if i == 0:
                i = 1
            else:
                i = 0

        return self.imgdata


    def verify_original(self,imgdata1,imgdata2):
        pass