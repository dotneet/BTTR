from bttr.lit_bttr import LitBTTR
from PIL import Image
from torchvision.transforms import ToTensor
import time
import torch

device = torch.device("cpu")
ckpt = './pretrained-2014.ckpt'
# ckpt = './self.ckpt'

start = time.time()
img_path = './example/18_em_1.bmp'
model = LitBTTR.load_from_checkpoint(ckpt, map_location=device)
print(f"load weight: {time.time() - start}")

start = time.time()
img = Image.open(img_path)
img = ToTensor()(img).to(device)
print(f"ToTensor: {time.time() - start}")

for i in range(0,1):
    hyp = model.beam_search(img)
    print(hyp)
    print(time.time() - start)


