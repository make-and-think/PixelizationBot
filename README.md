# PixelizationBot

PixelizationBot is a Telegram bot that allows you to pixelate images. It uses machine learning models to process images and provides a convenient interface for interaction via Telegram.

## Demo:
[@pixelizationaibot](https://t.me/pixelizationaibot)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/PixelizationBot.git
   cd PixelizationBot
   ```

2. **Install the required dependencies:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Download the models:**
   
   Our software download models automatically from [HuggingFace 🤗](https://huggingface.co/ashleykleynhans/pixelization/tree/main)
   
   But you can this yourself:
   Download all three models from the table and place them into the `/models` directory inside the extension.

   | URL | Filename |
   |-----|----------|
   | [Model 1 Link](https://drive.google.com/file/d/1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM/view?usp=sharing) | pixelart_vgg19.pth |
   | [Model 2 Link](https://drive.google.com/file/d/17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_/view?usp=sharing) | alias_net.pth |
   | [Model 3 Link](https://drive.google.com/file/d/1i_8xL3stbLWNF4kdQJ50ZhnRFhSDh3Az/view?usp=sharing) | 160_net_G_A.pth |

   Alternative download link for models: 
   - [Pixeldrain](https://pixeldrain.com/u/QfmACJAG)
   - [HuggingFace 🤗](https://huggingface.co/ashleykleynhans/pixelization/tree/main)
   

## Configuration Setup

1. **Create a configuration file:**

Copy `config/config.yml.example` to `config/config.yml` and edit it according to your needs.
 

`cp config/config.yml.example config/config.yml`   


2. **Configuration parameters:**
   - **Bot config:**
     - `API_ID` and `API_HASH`: Your Telegram application identifiers.
     - `API_TOKEN`: Your bot's token.
   - **Model config:**
     - `NETG_PATH`, `ALIASNET_PATH`, `VGG19_PATH`: path where place models
     - `REFERENCE_PATH`: path where stored coloro scheme image reference
   - **Worker config:**
     - `FORCE_USE_CPU`: Force to use CPU (if you have cuda card in your computer)
     - `NUM_PROCESS`: Number of workers for image processing.
     - `NUM_TORCH_THREADS`: How many CPU cores will be used.
     - `MODEL_KEEP_ALIVE_SECONDS`: Time in seconds for which models remain loaded during inactivity.
   - **Queue config:**
     - `SLOTS_QUANTITY`: How many one user can send pictures in queue.
     - `DELAY_STATUS`: How often we need update status in queue message

   Also you can use flag `--config` when start script, and set custom path for your configuration
   ```bash
   python3 main.py # looking in work folder by path config/config.yml
   
   python3 main.py --config /home/foxgirl/config/config.yml # take config by custom path
   ```

## Example systemd Service

```systemd
[Unit]
Description=Pixelization ML model Telegram bot.

[Service]
Restart=on-abort
Type=simple
ExecStart=PixelizationBot/venv/bin/python3 main.py # Set path to 
WorkingDirectory=PixelizationBot #Path where installed PixelizationBot

[Install]
WantedBy=default.target
```

# Credits
- [txlyre](https://github.com/txlyre) - prototype demo
- [Th3ro](https://github.com/Th3roo)  - memory leak fix and support with another developing
- [Taruu](https://github.com/Taruu)   - Idea author and developing 

## Original repo project:
 - https://github.com/WuZongWei6/Pixelization


# TODO 
 - Make better queue system (always can be better)
 - Remade Model network from scrap
 - make custom lib for better conversation