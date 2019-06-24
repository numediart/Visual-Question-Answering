import torchvision.models as models
import torch.nn as nn
from torch.autograd import Variable
import torch
from os import listdir
from os.path import isfile, join

from torchvision import transforms
resize = 128
crop = 128
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

# class NormalizeInverse(torchvision.transforms.Normalize):
#     """
#     Undoes the normalization and returns the reconstructed images in the input domain.
#     """
#
#     def __init__(self, mean, std):
#         mean = torch.as_tensor(mean)
#         std = torch.as_tensor(std)
#         std_inv = 1 / (std + 1e-7)
#         mean_inv = -mean * std_inv
#         super().__init__(mean=mean_inv, std=std_inv)
#
#     def __call__(self, tensor):
#         return super().__call__(tensor.clone())
#
#
# _transforms=[]
# if resize is not None:
#     _transforms.append(transforms.Resize(resize))
# if crop is not None:
#     _transforms.append(transforms.CenterCrop(crop))
# # _transforms.append(transforms.ToPILImage())
# _transforms.append(
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]))
# _transforms.append(transforms.ToTensor())
#
# transform = transforms.Compose(_transforms)
#
#
#
#
# with open('COCO_train2014_000000010130.jpg', 'rb') as f:
#     img = Image.open(f).convert('RGB')
#     img.show()
#     x = transform(img)
#
#
# # n = NormalizeInverse(mean=[0.485, 0.456, 0.406],
# #                          std=[0.229, 0.224, 0.225])
# # x = n(x)
# v = transforms.ToPILImage()
# x = v(x)
# x.show()



#################################"


# x.show()
#
#
# x.save("here.jpg","JPEG", quality=100)

# v = transforms.ToTensor()
# print(v(x))
#
# with open("here.jpg", 'rb') as f:
#     img = Image.open(f).convert('RGB')
#
# x = (img)
# print(v(x))

# plt.imshow(x.permute(1,2,0))
# plt.show()


#
#
model_ft = models.resnet50(pretrained=True)
model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
out = nn.Sequential(*list(model_ft.children())).cuda()
for param in out.parameters():
    param.requires_grad = False

x = Variable(torch.rand((2, 3, 224, 224))).cuda()

for block in out:
    x = block(x)
    print(x.shape)

