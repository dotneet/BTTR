from bttr.lit_bttr import LitBTTR
from PIL import Image
from torchvision.transforms import ToTensor

ckpt = './pretrained-2014.ckpt'
img_path = './example/18_em_1.bmp'
model = LitBTTR.load_from_checkpoint(ckpt)
img = Image.open(img_path)
img = ToTensor()(img)
hyp = model.beam_search(img)
print(hyp)

