
import numpy as np
def MyMedianFilter(Im):
    def MedianFiler(img):
        [w, h] = img.shape
        out = np.zeros((w + 2, h + 2))
        out[1:w + 1, 1:h + 1] = img  # 此处自己创建的out与img类型有差别
        for i in range(w):
            if i % 10 == 0:
                print(i)
            for j in range(h):
                if 1 < i < w and 1 < j < h:
                    temp = img[i - 1:i + 2, j - 1:j + 2]
                    med = np.median(temp)
                    out[i, j] = med
        print('function finished')
        return out

    size = Im.shape
    AfterMedFilter = MedianFiler(Im)
    AfterMedFilter = AfterMedFilter.astype(np.uint8)