import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.logic.networks import define_G

# Constants
NET_G_PATH = "models/160_net_G_A.pth"
ALIAS_NET_PATH = "models/alias_net.pth"
REFERENCE_PATH = "reference.png"

class PixelizationModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G_A_net = None
        self.alias_net = None
        self.ref_t = None

    def load(self):
        with torch.no_grad():
            self._load_networks()
            self._load_reference_image()

    def _load_networks(self):
        self.G_A_net = self._load_network(NET_G_PATH, "c2pGen")
        self.alias_net = self._load_network(ALIAS_NET_PATH, "antialias")

    def _load_network(self, path, net_type):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Network file not found: {path}")
        
        net = define_G(3, 3, 64, net_type, "instance", False, "normal", 0.02, [0])
        state = torch.load(path, map_location=self.device, weights_only=True)
        net.load_state_dict(state)
        return net

    def _load_reference_image(self):
        if not os.path.exists(REFERENCE_PATH):
            raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")
        
        ref_img = Image.open(REFERENCE_PATH).convert('L')
        self.ref_t = process_image(greyscale(ref_img)).to(self.device)

    def pixelize(self, input_path, output_path, pixel_size=4):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        with torch.no_grad():
            input_img = Image.open(input_path).convert('RGB')
            input_tensor = process_image(input_img, pixel_size).to(self.device)
            output_tensor = self.alias_net(self.G_A_net(input_tensor, self.ref_t))
            save_image(output_tensor, output_path, pixel_size)

def greyscale(img):
    gray = np.array(img.convert('L'))
    return Image.fromarray(np.stack((gray,)*3, axis=-1))

def process_image(img, pixel_size=4):
    img = resize_image(img, pixel_size)
    img = crop_image(img)
    return transform_image(img)

def resize_image(img, pixel_size):
    return img.resize((img.width * 4 // pixel_size, img.height * 4 // pixel_size))

def crop_image(img):
    ow, oh = img.size
    nw = int(round(ow / 4) * 4)
    nh = int(round(oh / 4) * 4)
    left = (ow - nw) // 2
    top = (oh - nh) // 2
    right = left + nw
    bottom = top + nh
    return img.crop((left, top, right, bottom))

def transform_image(img):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return trans(img)[None, :, :, :]

def save_image(tensor, file_path, pixel_size=4):
    img = tensor.data[0].cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((img.size[0]//4, img.size[1]//4), resample=Image.Resampling.NEAREST)
    img = img.resize((img.size[0]*pixel_size, img.size[1]*pixel_size), resample=Image.Resampling.NEAREST)
    img.save(file_path)

def main():
    if len(sys.argv) < 4:
        print("Usage: python pixelization.py <input_image> <output_image> <pixel_size>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    pixel_size = int(sys.argv[3])

    try:
        model = PixelizationModel()
        model.load()
        model.pixelize(input_path, output_path, pixel_size)
        print(f"Pixelization complete. Output saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()