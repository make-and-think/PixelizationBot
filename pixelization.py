import gc
import sys
import os
import logging
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import colorsys
from typing.io import BinaryIO

from models.logic.networks import define_G
from config.config import config

NETG_PATH = config.NETG_PATH
ALIASNET_PATH = config.ALIASNET_PATH
REFERENCE_PATH = config.REFERENCE_PATH

if config.__dict__.get("NUM_PROCESS"):
    torch.set_num_threads(1)

# TODO MOVE to config file
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pixelization.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)


# def save(tensor, file: BinaryIO, pixel_size=4, upscale_after=True, original_img=None, copy_hue=False, copy_sat=False):
#     img = to_image(tensor, pixel_size, upscale_after, original_img, copy_hue, copy_sat)
#     img.save(file)
#     logger.info(f"Image saved to {file}")


class PixelizationModel:
    def __init__(self, netG_path=NETG_PATH, aliasnet_path=ALIASNET_PATH, reference_path=REFERENCE_PATH):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

        self.netG_path = netG_path
        self.aliasnet_path = aliasnet_path
        self.reference_path = reference_path
        self.G_A_net = None
        self.alias_net = None
        self.ref_t = None

    @staticmethod
    def process(input_img: Image, pixel_size=4) -> torch.Tensor:
        input_img = input_img.resize((input_img.width * 4 // pixel_size, input_img.height * 4 // pixel_size))

        ow, oh = input_img.size
        nw = int(round(ow / 4) * 4)
        nh = int(round(oh / 4) * 4)

        left = (ow - nw) // 2
        top = (oh - nh) // 2
        right = left + nw
        bottom = top + nh

        input_img = input_img.crop((left, top, right, bottom))

        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        return transformer(input_img)[None, :, :, :]

    @staticmethod
    def rgb_to_hsv_array(rgb):
        r, g, b = rgb[..., 0] / 255.0, rgb[..., 1] / 255.0, rgb[..., 2] / 255.0
        hsv = np.array([colorsys.rgb_to_hsv(r[i], g[i], b[i]) for i in range(len(r))])
        return hsv

    @staticmethod
    def load_model_weights(model, weights_path, device):
        logger.info(f"Loading model weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model

    def color_image(self, img: Image.Image, original_img: Image.Image, copy_hue, copy_sat):
        """copy original hue and saturation"""
        img = img.convert("RGB")
        original_img = original_img.convert("RGB")

        img_data = np.array(img)
        original_data = np.array(original_img)

        original_hsv = self.rgb_to_hsv_array(original_data.reshape(-1, 3))
        img_hsv = self.rgb_to_hsv_array(img_data.reshape(-1, 3))

        if copy_hue:
            img_hsv[:, 0] = original_hsv[:, 0]
        if copy_sat:
            img_hsv[:, 1] = original_hsv[:, 1]

        rgb = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in img_hsv])
        img_data = (rgb * 255).astype(np.uint8).reshape(img_data.shape)

        return Image.fromarray(img_data)

    def to_image(self, tensor, pixel_size, upscale_after, original_img, copy_hue, copy_sat):
        img = tensor.squeeze().cpu().numpy()
        img = ((img + 1) / 2 * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        img = Image.fromarray(img)
        img = img.resize((img.width // 4, img.height // 4), resample=Image.Resampling.NEAREST)

        if copy_hue or copy_sat:
            original_img = original_img.resize(img.size, resample=Image.Resampling.NEAREST)
            img = self.color_image(img, original_img, copy_hue, copy_sat)

        if upscale_after:
            img = img.resize((img.width * pixel_size, img.height * pixel_size), resample=Image.Resampling.NEAREST)

        return img

    def unload(self):
        logger.info("Unload models...")
        del self.G_A_net
        del self.alias_net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.G_A_net = None
        self.alias_net = None

    def load(self):
        logger.info("Loading models...")
        with torch.no_grad():
            self.G_A_net = define_G(3, 3, 64, "c2pGen", "instance", False, "normal", 0.02,
                                    [0] if torch.cuda.is_available() else [])

            self.alias_net = define_G(3, 3, 64, "antialias", "instance", False, "normal", 0.02,
                                      [0] if torch.cuda.is_available() else [])

            self.G_A_net = self.load_model_weights(self.G_A_net, self.netG_path, self.device)
            self.alias_net = self.load_model_weights(self.alias_net, self.aliasnet_path, self.device)

            ref_img = Image.open(self.reference_path).convert('L')

            gray = np.array(ref_img.convert('L'))
            gray_tmp = np.expand_dims(gray, axis=2)
            gray_tmp = np.concatenate((gray_tmp, gray_tmp, gray_tmp), axis=-1)
            greyscale_image = Image.fromarray(gray_tmp)

            self.ref_t = self.process(greyscale_image).to(self.device)

        logger.info("Models loaded successfully")

    def pixelize(self, input_img: Image.Image, pixel_size=4, upscale_after=True, copy_hue=False,
                 copy_sat=False) -> Image.Image:
        logger.info(f"Pixelizing image with pixel size {pixel_size}")

        with torch.no_grad():
            original_img = input_img.convert('RGB')
            in_t = self.process(original_img, pixel_size).to(self.device)
            out_t = self.alias_net(self.G_A_net(in_t, self.ref_t))

        logger.info("Start start to_image")
        self.unload()
        self.load()
        return self.to_image(out_t, pixel_size, upscale_after, original_img, copy_hue, copy_sat)


def main():
    if len(sys.argv) < 4:
        logger.error(
            "Usage: python pixelization.py <input_image> <output_image> <pixel_size> [--upscale-after] [--copy-hue] [--copy-sat]")
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

    # Загружаем изображение и обрабатываем его
    original_img = Image.open(input_image)
    processed_img = model.pixelize(original_img, pixel_size, upscale_after, copy_hue, copy_sat)

    processed_img.save(output_image)
    logger.info(f"Image saved to {output_image}")


if __name__ == "__main__":
    main()
