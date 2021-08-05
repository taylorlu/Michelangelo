from logging import root
import os, cv2
import numpy as np

class Loader:
    def __init__(self, root_path) -> None:
        self.root_path = root_path
    
    def get_alpha_data(self, file_path):
        if (self.root_path):
            file_path = os.path.join(self.root_path, file_path)


class VideoMatte240KLoader():
    def __init__(self, root_path, pha='pha', fgr='fgr') -> None:
        self.root_path = root_path
        self.pha = pha
        self.fgr = fgr

    def get_data(self, file_path):
        if (self.root_path):
            pha_file_path = os.path.join(self.root_path, 'train', self.pha, file_path)
            fgr_file_path = os.path.join(self.root_path, 'train', self.fgr, file_path)

        pha_cap = cv2.VideoCapture(pha_file_path)
        fgr_cap = cv2.VideoCapture(fgr_file_path)

        flag = True
        while(True):
            ret, pha_frame = pha_cap.read()
            ret2, fgr_frame = fgr_cap.read()
            if(ret is not True or ret2 is not True):
                print(ret, ret2)
                break
            else:
                gray = pha_frame[..., 0]
                if(gray.shape[0]!=1080 and flag is True):
                    print(file_path)
                    flag = False


class RWP636Loader():
    def __init__(self, root_path) -> None:
        self.root_path = root_path

    def get_data(self, file_path):
        if (self.root_path):
            pha_file_path = os.path.join(self.root_path, 'train', self.pha, file_path)




if(__name__=='__main__'):

    files = os.listdir(r'E:\CVDataset\VideoMatte240K\train\pha')
    for f in files:
        loader = VideoMatte240KLoader(r'E:\CVDataset\VideoMatte240K')
        loader.get_data(f)
