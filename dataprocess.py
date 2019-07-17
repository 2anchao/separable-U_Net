from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class Mydataset(Dataset):
    CLASSES = [0, 1, 2]
    def __len__(self):
        return len(self.ids)
    def __init__(self,images_dir:str,masks_dir:str,nb_classes,classes=None,transform=None):
        super().__init__()
        self.class_values = [self.CLASSES.index(cls) for cls in classes]
        self.nb_classes=nb_classes
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '.npy') for image_id in self.ids]
        self.transform=transform

    def __getitem__(self, i):
        image = Image.open(self.images_fps[i])
        mask = np.load(self.masks_fps[i])
        mask[mask > self.nb_classes - 1] = 0
        #mask=Image.fromarray(mask)
        #change=transforms.Resize((192,256),2)
        #mask=change(mask)
        #mask=np.array(mask)

        if self.transform is not None:
            image = self.transform(image)
        return image, mask

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

