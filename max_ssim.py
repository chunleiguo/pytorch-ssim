import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import PIL.Image as pil_image

import numpy as np

npImg1 = pil_image.open("einstein.png")

img1 = torch.from_numpy(np.array(npImg1)).float().unsqueeze(0).unsqueeze(0)/255.0

img2 = torch.rand(img1.size())

if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()


img1 = Variable( img1,  requires_grad=False)
img2 = Variable( img2, requires_grad = True)


# Functional: pytorch_ssim.ssim(img1, img2, window_size = 11, size_average = True)
ssim_value = pytorch_ssim.ssim(img1, img2).item()
print("Initial ssim:", ssim_value)

# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
ssim_loss = pytorch_ssim.SSIM()

optimizer = optim.Adam([img2], lr=0.01)

while ssim_value < 0.95:
    optimizer.zero_grad()
    ssim_out = -ssim_loss(img1, img2)
    ssim_value = - ssim_out.item()

    print(ssim_value)
    ssim_out.backward()
    optimizer.step()


img2 = np.floor(img2.detach().numpy().squeeze(0).squeeze(0)*255 + 0.5)
img2 = np.clip(img2, 0.0, 255).astype(np.uint8)
out = pil_image.fromarray(img2)
out.save('out.png')