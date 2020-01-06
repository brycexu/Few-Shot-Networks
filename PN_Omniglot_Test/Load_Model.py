# Author : Bryce Xu
# Time : 2019/11/25
# Function: 康康模型里面的东西

import torch

model = torch.load('Result/55.pth', map_location='cpu')

print(model)
