# Phone Driver

A Python-based mobile automation agent that uses Qwen3-VL vision-language models to understand and interact with Android devices through visual analysis and ADB commands.

<p align="center">
  <img src="Images/PhoneDriver.png" width="600" alt="Phone Driver Demo">
</p>

## Features

- 🤖 **Vision-powered automation**: Uses Qwen3-VL to visually understand phone screens
- 📱 **ADB integration**: Controls Android devices via ADB commands
- 🎯 **Natural language tasks**: Describe what you want in plain English
- 🖥️ **Web UI**: Built-in Gradio interface for easy control
- 📊 **Real-time feedback**: Live screenshots and execution logs
- 🌐 **Flexible providers**: Use local models or OpenAI-compatible API endpoints

## Requirements

### Core Requirements (All Users)
- Python 3.10+
- Android device with USB debugging & Developer Mode enabled
- ADB (Android Debug Bridge) installed

### Additional Requirements (Local Model Users Only)
- **GPU with sufficient VRAM** (Tested on 24gb GPU with Qwen3-VL-8B Model)
- **PyTorch with CUDA support**
- **Transformers library**
- The Repo is set to use the Dense Qwen3-VL 4B/8B Model which performs very well. To swap to an MoE model, see the configuration section below

### API Provider Users
- **No GPU or NVIDIA dependencies required** - works with external API endpoints (OpenAI, Azure, vLLM, etc.) 

## Installation

### 1. Install ADB

**Linux/Ubuntu:**
```bash
sudo apt update
sudo apt install adb
```
### 2. Clone Repo & Install Python Dependencies

```bash
git clone https://github.com/OminousIndustries/PhoneDriver.git
cd PhoneDriver
```
Create a Virtual Enviornment

```bash
python -m venv phonedriver
source phonedriver/bin/activate
```

#### Option A: For API Provider (OpenAI-compatible APIs)

If you plan to use external API providers (recommended for users without NVIDIA GPUs), install minimal dependencies:

```bash
pip install pillow gradio requests
```

#### Option B: For Local Model Provider (Requires NVIDIA GPU)

If you plan to run models locally on your GPU, install all dependencies including PyTorch and transformers:

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers from source (required for Qwen3-VL)
pip install git+https://github.com/huggingface/transformers
# pip install transformers==4.57.0 # currently, V4.57.0 is not released

# Install other requirements
pip install pillow gradio qwen_vl_utils requests
```

### 3. Connect Your Device

1. Enable USB debugging on your Android device (Settings → Developer Options)
2. Connect via USB
3. Verify connection:
```bash
adb devices
```
You should see your device listed.

## Configuration

### Provider Selection

PhoneDriver supports two modes:

#### 1. Local Model (Requires NVIDIA GPU)

**Requirements:** PyTorch, transformers, CUDA-enabled GPU

Edit `config.json` to use a local model:

```json
{
  "provider": "local",
  "model_name": "Qwen/Qwen3-VL-8B-Instruct",
  "use_flash_attention": false,
  ...
}
```

Available local models:
- `Qwen/Qwen3-VL-4B-Instruct` - Smaller, faster
- `Qwen/Qwen3-VL-8B-Instruct` - Recommended balance
- `Qwen/Qwen3-VL-30B-A3B-Instruct` - Highest quality

#### 2. OpenAI-Compatible API (No GPU Required)

**Requirements:** Only basic Python libraries (pillow, gradio, requests)

Use any OpenAI-compatible API endpoint:

```json
{
  "provider": "api",
  "model_name": "gpt-4-vision-preview",
  "api_base_url": "https://api.openai.com/v1",
  "api_key": "sk-your-api-key-here",
  ...
}
```

This works with:
- OpenAI API
- Azure OpenAI
- Local inference servers (vLLM, Text Generation WebUI, etc.)
- Any OpenAI-compatible endpoint

### MoE Models (Local Only)

If you want to try a Qwen3 MoE model, you need to change the import in `qwen_vl_agent.py`:

```python
#from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  - Comment this import out
# Uncomment the import below for the MoE Variants!!!
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
```

And update the model initialization (line ~92):

```python
self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
```

### Screen Resolution

The agent can auto-detect your device resolution from the Web UI settings tab, but you can manually configure it in `config.json`.

```json
{
  "screen_width": 1080,
  "screen_height": 2340,
  ...
}
```

To get your device resolution, with the device connected to your computer type the following in the terminal: 
```bash
adb shell wm size
```

## Usage

### Web UI (Recommended)

Launch the Gradio interface:

```bash
python ui.py
```

Navigate to `http://localhost:7860` and enter tasks like:
- "Open Chrome"
- "Search for weather in New York"
- "Open Settings and enable WiFi"

### Command Line

```bash
python phone_agent.py "your task here"
```

Example:
```bash
python phone_agent.py "Open the camera app"
```

## How It Works

1. **Screenshot Capture**: Takes a screenshot of the phone via ADB
2. **Visual Analysis**: Qwen3-VL analyzes the screen to understand UI elements
3. **Action Planning**: Determines the best action to take (tap, swipe, type, etc.)
4. **Execution**: Sends ADB commands to perform the action
5. **Repeat**: Continues until task is complete or max cycles reached

## Configuration Options

Key settings in `config.json`:

### Provider Settings
- `provider`: Provider type - "local" or "api" (default: "local")
- `model_name`: Model identifier (HuggingFace path for local, model name for API)
- `api_base_url`: API endpoint URL (required for API provider)
- `api_key`: API authentication key (required for API provider)

### Model Parameters
- `temperature`: Model creativity (0.0-1.0, default: 0.1)
- `max_tokens`: Max response length (default: 512)
- `use_flash_attention`: Enable Flash Attention 2 for faster local inference

### Execution Settings
- `step_delay`: Wait time between actions in seconds (default: 1.5)
- `max_retries`: Maximum retry attempts (default: 3)
- `screen_width`: Device screen width in pixels
- `screen_height`: Device screen height in pixels
- `enable_visual_debug`: Save annotated screenshots for debugging

## Troubleshooting

**Device not detected:**
- Ensure USB debugging is enabled
- Run `adb devices` to verify connection
- Try `adb kill-server && adb start-server`

**Wrong tap locations:**
- Auto-detect resolution in Settings tab of UI
- Or manually verify with `adb shell wm size`

**Model loading errors:**
- Ensure you have sufficient VRAM
- Try the 8B model for lower memory requirements
- Check that transformers is installed from source

**Out of memory:**
- Use the 8B model instead of 30B
- Reduce `max_tokens` in config
- Close other applications using GPU memory

## License

Apache License 2.0 - see LICENSE file for details

## Acknowledgments

- Built with [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) by Alibaba Cloud
- Uses [Gradio](https://gradio.app/) for the web interface
