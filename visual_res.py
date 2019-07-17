
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import imgaug as ia
from PIL import Image


def data_process(x):
    x = np.array(x, dtype='float32') / 255
    x1 = np.expand_dims(x, 0)  #z在位置0加一个维度
    x = np.transpose(x1, (0, 3, 1, 2))
    x = torch.from_numpy(x)
    mean = [0.517446, 0.360147, 0.310427]
    std = [0.061526, 0.049087, 0.041330]
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)
    return x.sub_(mean[:, None, None]).div_(std[:, None, None])

def sigmoid(x):
    s = 1 / (1 + np.exp(x))
    return s

def visulize_result(image, pred, logits=False):
    if logits:
        pred = sigmoid(pred)

    mask = (np.round(pred)).astype(np.uint8)
    mask = mask.squeeze()
    segmap = ia.SegmentationMapOnImage(mask, shape=mask.shape, nb_classes=3)
    colors = [(0, 0, 255), (255, 255, 255), (255, 0, 255)]
    img_add_mask = segmap.draw_on_image(image, colors=colors, alpha=0.4)

    cells = [image, img_add_mask]
    grid_image = ia.draw_grid(cells, cols=2)

    return img_add_mask, grid_image



if __name__ == '__main__':
    Vis = True
    model = torch.load('0.8065_model.pth')
    imgspath = r"/home/anchao/桌面/u_net/data"
    imgs = os.listdir(imgspath)
    Ltime = []
    i=0
    for img in imgs:
        i+=1
        print("...%d"%i)
        image_path=os.path.join(imgspath, img)
        image = plt.imread(image_path)
        x = data_process(image)
        start_time = time.time()
        y = model(x.cuda().float())
        y = torch.nn.Softmax2d()(y)
        end_time = time.time()
        numpy_y = y.cpu().detach().numpy()[0, ...]
        y1 = np.argmax(numpy_y, axis=0)
        y1=np.array(y1,dtype='float32')
        print("current img infer cost time:{:.2f} ms".format((end_time - start_time) * 1000))
        Ltime.append((end_time - start_time) * 1000)
        if Vis:
            plt.figure(figsize=(12,4),dpi=240)

            plt.subplot(1, 2, 1)
            plt.title("original image",loc='left')
            plt.axis(False)
            plt.imshow(image)

            plt.subplot(1, 2, 2)
            plt.title("predict_result",loc='left')
            plt.axis(False)
            predict_result=visulize_result(image, y1)[0]
            plt.imshow(predict_result)

            os.makedirs("save_visual", exist_ok=True)
            plt.savefig(os.path.join("save_visual", img.split('.')[0] + '.jpg'))#,bbox_inches='tight'
    print("Mean Time:{:.2f} ms".format(sum(Ltime[1:]) / (len(Ltime) - 1)))
