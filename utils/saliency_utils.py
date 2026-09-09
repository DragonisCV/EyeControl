import cv2
import numpy as np
from models.TranSalNet.utils.data_process import preprocess_img, postprocess_img
import torch
from torchvision import transforms


def get_saliency_map_transalnet(model,img,img_path=None):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = preprocess_img(img_path,img=img) # padding and resizing input image into 384x288
    img = np.array(img)/255.
    img = np.expand_dims(np.transpose(img,(2,0,1)),axis=0)
    img = torch.from_numpy(img)
    img = img.to(next(model.parameters()).device,dtype=torch.float32)
    pred_saliency = model(img)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())

    pred_saliency = postprocess_img(pic, img_path,org=img_gray) # restore the image to its original size as the result
    # print(pred_saliency.shape)
    # pred_saliency = np.stack([pred_saliency,pred_saliency,pred_saliency],axis=-1)
    return pred_saliency
