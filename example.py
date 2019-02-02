import torch
import models
import os
import numpy as np
from data_loader import BSDS_RCFLoader
from torch.utils.data import DataLoader
from PIL import Image

resume = 'ckpt/10001.pth'
folder = 'results/test-tmp/'

model = models.resnet101(pretrained=True).cuda()
model.eval()

#resume..
checkpoint = torch.load(resume)
model.load_state_dict(checkpoint)

train_dataset = BSDS_RCFLoader(split="train")
train_loader = DataLoader(
    train_dataset, batch_size=1,
    num_workers=1, drop_last=False, shuffle=False)

with torch.no_grad():
    for i, (image, label) in enumerate(train_loader):
        image, label = image.cuda(), label.cuda()
        outs = model(image, label.size()[2:])
        label[label == 2] = 0

        outs.append(label)
        outs.append(image)

        idx = 0
        print('working on .. {}'.format(i))
        for result in outs:
            idx += 1
            result = result.squeeze().detach().cpu().numpy()
            if len(result.shape) == 3:
                result = result.transpose(1, 2, 0).astype(np.uint8)
                result = result[:, :, [2, 1, 0]]
                Image.fromarray(result).save(os.path.join(folder, '{}-img.jpg'.format(i)))
            else:
                result = (result * 255).astype(np.uint8)
                Image.fromarray(result).save(os.path.join(folder, '{}-{}.png'.format(i, idx)))
        if i >= 20:
            break
    print('finished.')

