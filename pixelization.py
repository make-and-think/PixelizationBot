import sys
import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import colorsys
from PIL import Image
from models.logic.networks import define_G

NETG_PATH = "models/160_net_G_A.pth"
ALIASNET_PATH = "models/alias_net.pth"
REFERENCE_PATH = "reference.png"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pixelization.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device


def load_model_weights(model, weights_path, device):
    logger.info(f"Loading model weights from: {weights_path}")
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


def rgb_to_hsv_array(rgb):
    r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(r[i], g[i], b[i]) for i in range(len(r))])
    return hsv


def hsv_to_rgb_array(hsv):
    rgb = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in hsv])
    return (rgb * 255).astype(np.uint8)


# copy original hue and saturation
def color_image(img, original_img, copy_hue, copy_sat):
    img = img.convert("RGB")
    original_img = original_img.convert("RGB")

    img_data = np.array(img)
    original_data = np.array(original_img)

    original_hsv = rgb_to_hsv_array(original_data.reshape(-1, 3))
    img_hsv = rgb_to_hsv_array(img_data.reshape(-1, 3))

    if copy_hue:
        img_hsv[:, 0] = original_hsv[:, 0]
    if copy_sat:
        img_hsv[:, 1] = original_hsv[:, 1]

    img_data = hsv_to_rgb_array(img_hsv).reshape(img_data.shape)

    return Image.fromarray(img_data)


def to_image(tensor, pixel_size, upscale_after, original_img, copy_hue, copy_sat):
    img = tensor.squeeze().cpu().numpy()
    img = ((img + 1) / 2 * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    img = img.resize((img.width // 4, img.height // 4), resample=Image.Resampling.NEAREST)

    if copy_hue or copy_sat:
        original_img = original_img.resize(img.size, resample=Image.Resampling.NEAREST)
        img = color_image(img, original_img, copy_hue, copy_sat)

    if upscale_after:
        img = img.resize((img.width * pixel_size, img.height * pixel_size), resample=Image.Resampling.NEAREST)

    return img


def save(tensor, file, pixel_size=4, upscale_after=True, original_img=None, copy_hue=False, copy_sat=False):
    img = to_image(tensor, pixel_size, upscale_after, original_img, copy_hue, copy_sat)
    img.save(file)
    logger.info(f"Image saved to {file}")


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
        logger.info("Loading models...")
        with torch.no_grad():
            self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02, [0] if torch.cuda.is_available() else [])
            self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02, [0] if torch.cuda.is_available() else [])

            self.G_A_net = load_model_weights(self.G_A_net, self.netG_path, self.device)
            self.alias_net = load_model_weights(self.alias_net, self.aliasnet_path, self.device)

            ref_img = Image.open(self.reference_path).convert('L')
            self.ref_t = process(greyscale(ref_img)).to(self.device)

        logger.info("Models loaded successfully")

    def pixelize(self, in_img, out_img, pixel_size=4, upscale_after=True, copy_hue=False, copy_sat=False):
        logger.info(f"Pixelizing image {in_img} with pixel size {pixel_size}")
        with torch.no_grad():
            original_img = Image.open(in_img).convert('RGB')
            in_t = process(original_img, pixel_size).to(self.device)

            out_t = self.alias_net(self.G_A_net(in_t, self.ref_t))

            save(out_t, out_img, pixel_size, upscale_after, original_img, copy_hue, copy_sat)

        logger.info(f"Pixelization completed for image {in_img}. Output saved to {out_img}")


def main():
    if len(sys.argv) < 4:
        logger.error("Usage: python pixelization.py <input_image> <output_image> <pixel_size> [--upscale-after] [--copy-hue] [--copy-sat]")
        sys.exit(1)

    input_image = sys.argv[1]
    output_image = sys.argv[2]
    pixel_size = int(sys.argv[3])

    upscale_after = '--upscale-after' in sys.argv
    copy_hue = '--copy-hue' in sys.argv
    copy_sat = '--copy-sat' in sys.argv

    if not os.path.isfile(input_image):
        logger.error(f"Input image '{input_image}' does not exist.")
        sys.exit(1)

    model = PixelizationModel()
    model.load()
    model.pixelize(input_image, output_image, pixel_size, upscale_after, copy_hue, copy_sat)


if __name__ == "__main__":
    main()
