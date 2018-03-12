import numpy as np
from skimage.measure import compare_ssim as ssim

class CheckerBoard:

    def __init__(self):
        pass

    def run(self, imgdata):
        self.imgdata = imgdata.copy()
        self.imgdata.setflags(write=1)

        i = 0
        for x in range(0, len(self.imgdata)):       # for each row
            for y in range(i, len(self.imgdata[x]), 2):
                self.imgdata[x][y] = [255,0,0,0]
            if i == 0:
                i = 1
            else:
                i = 0

        return self.imgdata

    def get_difference(self,imgdata1,imgdata2):
        #pairs = zip(imgdata1,imgdata2)
        #dif = sum(abs(c1-c2) for p1,p2 in pairs for c1,c2 in zip(p1,p2))
         
        #error = np.sum((imgdata1.astype("float") - imgdata2.astype("float")) ** 2)
        #error /= float(imgdata1.shape[0] * imgdata1.shape[1])
        
        error = ssim(imgdata1,imgdata2,multichannel=True)

        #error = np.mean(imgdata1 != imgdata2)
        print('\nSSIM: %s' % error)
        #ncomponents = len(imgdata1) * 3
        #print ("\nDifference %s%%" % round((dif / 255.0 * 100) / ncomponents, 4))