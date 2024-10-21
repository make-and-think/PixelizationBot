import sys
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import colorsys

from PIL import Image
from models.logic.networks import define_G

# Пути к файлам моделей и эталонному изображению
NETG_PATH = "models/160_net_G_A.pth"
ALIASNET_PATH = "models/alias_net.pth"
REFERENCE_PATH = "reference.png"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_weights(model, weights_path, device):
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model


def greyscale(img):
    gray = np.array(img.convert('L'))
    tmp = np.expand_dims(gray, axis=2)
    tmp = np.concatenate((tmp, tmp, tmp), axis=-1)
    return Image.fromarray(tmp)


def process(img, pixel_size=4):
    img = img.resize((img.width * 4 // pixel_size, img.height * 4 // pixel_size))

    ow, oh = img.size
    nw = int(round(ow / 4) * 4)
    nh = int(round(oh / 4) * 4)

    left = (ow - nw) // 2
    top = (oh - nh) // 2
    right = left + nw
    bottom = top + nh

    img = img.crop((left, top, right, bottom))

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return trans(img)[None, :, :, :]

# copy original hue and saturation
def color_image(img, original_img, copy_hue, copy_sat):
    img = img.convert("RGB")
    original_img = original_img.convert("RGB")

    colored_img = Image.new("RGB", img.size)

    for x in range(img.width):
        for y in range(img.height):
            pixel = original_img.getpixel((x, y))
            r, g, b = pixel
            original_h, original_s, original_v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

            pixel = img.getpixel((x, y))
            r, g, b = pixel
            h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)

            r, g, b = colorsys.hsv_to_rgb(original_h if copy_hue else h, original_s if copy_sat else s, v)
            colored_img.putpixel((x, y), (int(r * 255), int(g * 255), int(b * 255)))

    return colored_img


def to_image(tensor, pixel_size, upscale_after, original_img, copy_hue, copy_sat):
    img = tensor.data[0].cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    width = img.size[0] // 4
    height = img.size[1] // 4
    img = img.resize((width, height), resample=Image.Resampling.NEAREST)

    if copy_hue or copy_sat:
        original_img = original_img.resize((width, height), resample=Image.Resampling.NEAREST)
        img = color_image(img, original_img, copy_hue, copy_sat)

    if upscale_after:
        img = img.resize((img.size[0] * pixel_size, img.size[1] * pixel_size), resample=Image.Resampling.NEAREST)

    return img


def save(tensor, file, pixel_size=4, upscale_after=True, original_img=None, copy_hue=False, copy_sat=False):
    img = to_image(tensor, pixel_size, upscale_after, original_img, copy_hue, copy_sat)
    img.save(file)


class PixelizationModel:
    def __init__(self, netG_path=NETG_PATH, aliasnet_path=ALIASNET_PATH, reference_path=REFERENCE_PATH):
        self.device = get_device()
        self.netG_path = netG_path
        self.aliasnet_path = aliasnet_path
        self.reference_path = reference_path
        self.G_A_net = None
        self.alias_net = None
        self.ref_t = None

    def load(self):
        with torch.no_grad():
            self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0] if torch.cuda.is_available() else [])
            self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [0] if torch.cuda.is_available() else [])

            self.G_A_net = load_model_weights(self.G_A_net, self.netG_path, self.device, weights_only=True)
            self.alias_net = load_model_weights(self.alias_net, self.aliasnet_path, self.device, weights_only=True)

            ref_img = Image.open(self.reference_path).convert('L')
            self.ref_t = process(greyscale(ref_img)).to(self.device)

    def pixelize(self, in_img, out_img, pixel_size=4, upscale_after=True, copy_hue=False, copy_sat=False):
        with torch.no_grad():
            original_img = Image.open(in_img).convert('RGB')
            in_t = process(original_img, pixel_size).to(self.device)

            out_t = self.alias_net(self.G_A_net(in_t, self.ref_t))

            save(out_t, out_img, pixel_size, upscale_after, original_img, copy_hue, copy_sat)


def main():
    if len(sys.argv) < 4:
        print("Usage: python pixelization.py <input_image> <output_image> <pixel_size> [--upscale-after] [--copy-hue] [--copy-sat]")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2]
    pixel_size = int(sys.argv[3])

    upscale_after = '--upscale-after' in sys.argv
    copy_hue = '--copy-hue' in sys.argv
    copy_sat = '--copy-sat' in sys.argv

    if not os.path.isfile(input_image):
        print(f"Input image '{input_image}' does not exist.")
        sys.exit(1)

    model = PixelizationModel()
    model.load()
    model.pixelize(input_image, output_image, pixel_size, upscale_after, copy_hue, copy_sat)


if __name__ == "__main__":
    main()