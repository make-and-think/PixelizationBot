# PixelizationBot

PixelizationBot is a Telegram bot that allows you to pixelate images. It uses machine learning models to process images and provides a convenient interface for interaction via Telegram.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/PixelizationBot.git
   cd PixelizationBot
   ```

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the models:**

   Download all three models from the table and place them into the `/models` directory inside the extension.

   | URL | Filename |
   |-----|----------|
   | [Model 1 Link](https://drive.google.com/file/d/1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM/view?usp=sharing) | pixelart_vgg19.pth |
   | [Model 2 Link](https://drive.google.com/file/d/17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_/view?usp=sharing) | alias_net.pth |
   | [Model 3 Link](https://drive.google.com/file/d/1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az/view?usp=sharing) | 160_net_G_A.pth |

   Alternative download link for models: [Pixeldrain](https://pixeldrain.com/u/QfmACJAG)

## Configuration Setup

1. **Create a configuration file:**

   Copy `config/config.example` to `config/config` and edit it according to your needs.

2. **Configuration parameters:**

   - `API_ID` and `API_HASH`: Your Telegram application identifiers.
   - `API_TOKEN`: Your bot's token.
   - `NUM_PROCESS`: Number of workers for image processing.
   - `MODEL_KEEP_ALIVE_SECONDS`: Time in seconds for which models remain loaded during inactivity.
   - `NUM_TORCH_THREADS`: How many CPU cores will be used.

## Example systemd Service


# Credits
* Original repo: https://github.com/WuZongWei6/Pixelization