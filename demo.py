import torch
import models
import cv2
import numpy as np
from PIL import Image

from data_loader import prepare_image_cv2

resume = 'ckpt/lr-0.01-iter-490000.pth'
img_path = './examples/all_layer/100007-img.jpg'
result_path = './examples/demo.png'

model = models.resnet101(pretrained=False).cuda()
model.eval()

#resume..
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)

original_img = np.array(cv2.imread(img_path), dtype=np.float32)
h, w, _ = original_img.shape

img = prepare_image_cv2(original_img)
img = torch.from_numpy(img).unsqueeze(0).cuda()

outs = model(img, (h, w))
result = outs[-1].squeeze().detach().cpu().numpy()

result = (result * 255).astype(np.uint8)
Image.fromarray(result).save(result_path)

