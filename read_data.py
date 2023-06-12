import os
import cv2
import numpy as np
from torch.utils.data import Dataset

def convert_BGR2Lab(img_bgr):

    img_bgr = img_bgr.astype(np.float32)/255.0
    # transform to lab
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    # normalize from -1 to 1
    img_lab[:, :, 0] = img_lab[:, :, 0]/50 - 1
    img_lab[:, :, 1] = img_lab[:, :, 1]/127
    img_lab[:, :, 2] = img_lab[:, :, 2]/127
    # transpose
    img_lab = img_lab.transpose((2, 0, 1))

    return img_lab

def convert_Lab2BGR(img_lab):
    # transpose back
    img_lab = img_lab.transpose((1, 2, 0))
    # transform back
    img_lab[:, :, 0] = (img_lab[:, :, 0] + 1)*50
    img_lab[:, :, 1] = img_lab[:, :, 1]*127
    img_lab[:, :, 2] = img_lab[:, :, 2]*127
    # transform to bgr
    img_bgr = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    # to int8
    img_bgr = (img_bgr*255.0).astype(np.uint8)

    return img_bgr

"""Read data"""
class CustomDataset(Dataset):
    def __init__(self,image_dir):
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):

        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (256,256), interpolation = cv2.INTER_CUBIC)

        return convert_BGR2Lab(resized_image)

    @classmethod
    def getDatasetPath(cls, data_path):

        datasets = dict()
        datasets['train'] = cls(image_dir=os.path.join(data_path,'train'))
        datasets['val'] = cls(image_dir=os.path.join(data_path,'val'))
        datasets['test'] = cls(image_dir=os.path.join(data_path,'test'))
        return datasets
