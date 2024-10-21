import sys

import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image

from models.logic.networks import define_G

netGPath = "models/160_net_G_A.pth"
aliasnetPath = "models/alias_net.pth"

referencePath = "reference.png"

class Model():
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.G_A_net = None
    self.alias_net = None
    self.ref_t = None

  def load(self):
    with torch.no_grad():
      self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0])
      self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [0])

      G_A_state = torch.load(netGPath, map_location=self.device, weights_only=True)
      self.G_A_net.load_state_dict(G_A_state)

      alias_state = torch.load(aliasnetPath, map_location=self.device, weights_only=True)
      self.alias_net.load_state_dict(alias_state)

      ref_img = Image.open(referencePath).convert('L')
      self.ref_t = process(greyscale(ref_img)).to(self.device)

  def pixelize(self, in_img, out_img, pixel_size=4):
    with torch.no_grad():
      in_img = Image.open(in_img).convert('RGB')
      in_t = process(in_img, pixel_size).to(self.device)

      out_t = self.alias_net(self.G_A_net(in_t, self.ref_t))

      save(out_t, out_img, pixel_size)

def greyscale(img):
  gray = np.array(img.convert('L'))
  tmp = np.expand_dims(gray, axis=2)
  tmp = np.concatenate((tmp, tmp, tmp), axis=-1)
  return Image.fromarray(tmp)

def process(img, pixel_size=4):
  img = img.resize((img.width * 4 // pixel_size, img.height * 4 // pixel_size))

  ow,oh = img.size

  nw = int(round(ow / 4) * 4)
  nh = int(round(oh / 4) * 4)

  left = (ow - nw)//2
  top = (oh - nh)//2
  right = left + nw
  bottom = top + nh

  img = img.crop((left, top, right, bottom))

  trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  return trans(img)[None, :, :, :]

def save(tensor, file, pixel_size=4):
  img = tensor.data[0].cpu().float().numpy()
  img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
  img = img.astype(np.uint8)
  img = Image.fromarray(img)
  img = img.resize((img.size[0]//4, img.size[1]//4), resample=Image.Resampling.NEAREST)
  img = img.resize((img.size[0]*pixel_size, img.size[1]*pixel_size), resample=Image.Resampling.NEAREST)
  img.save(file)

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("Usage: python pixelization.py <input_image> <output_image> <pixel_size>")
    sys.exit(1)
  m = Model()
  m.load()
  m.pixelize(sys.argv[1], sys.argv[2], int(sys.argv[3]))

