# qwen_vl_agent.py
import json
import logging
import re
import base64
import io
from typing import Any, Dict, List, Optional

from PIL import Image
import warnings

# Conditional imports for local model support (requires NVIDIA dependencies)
# These are only imported when provider='local'
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor  # NOT MoeFor
    #from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor - This is only for the MoE Variants!!!
except ImportError:
    Qwen3VLForConditionalGeneration = None
    AutoProcessor = None

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

try:
    import requests
except ImportError:
    requests = None

# To supress these warnings you can uncomment the following two lines
# warnings.filterwarnings('ignore', message='.*Flash Efficient attention.*')
# warnings.filterwarnings('ignore', message='.*Mem Efficient attention.*')


class QwenVLAgent:
    """
    Vision-Language agent using Qwen3-VL for mobile GUI automation.
    Supports both local models and OpenAI-compatible API endpoints.
    Uses the official mobile_use function calling format.
    """
    
    # Constants
    MIN_SWIPE_DISTANCE = 0.05  # Minimum swipe distance as fraction of screen

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device_map: str = "auto",
        dtype: Optional[Any] = None,
        use_flash_attention: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 512,
        provider: str = "local",
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the Qwen3-VL agent.
        
        Args:
            model_name: Model name (HuggingFace for local, model ID for API)
            device_map: Device mapping for local models
            dtype: Data type for local models
            use_flash_attention: Enable Flash Attention 2 for local models
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            provider: "local" for local models, "api" for OpenAI-compatible API
            api_base_url: Base URL for API provider (e.g., "https://api.openai.com/v1")
            api_key: API key for authentication
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider = provider
        self.api_base_url = api_base_url
        self.api_key = api_key
        
        # Model and processor (only for local)
        self.model = None
        self.processor = None
        
        # System prompt used by both local and API providers
        self.system_prompt = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\n* The screen's resolution is 999x999.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `click`: Click the point on the screen with coordinate (x, y).\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\n* `type`: Input the specified text into the activated input box.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "swipe", "type", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to click. Required only by `action=click` and `action=swipe`. Range: 0-999.", "type": "array"}, "coordinate2": {"description": "(x, y): The end coordinates for swipe. Required only by `action=swipe`. Range: 0-999.", "type": "array"}, "text": {"description": "Required only by `action=type`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- Do not output anything else outside those three parts.
- If finishing, use action=terminate in the tool call.
- For each function call, there must be an "action" key in the "arguments" which denote the type of the action.
- Coordinates are in 999x999 space where (0,0) is top-left and (999,999) is bottom-right."""

        if self.provider == "local":
            self._init_local_model(model_name, device_map, dtype, use_flash_attention)
        elif self.provider == "api":
            self._init_api_provider()
        else:
            raise ValueError(f"Unknown provider: {provider}. Must be 'local' or 'api'")
    
    def _init_local_model(
        self,
        model_name: str,
        device_map: str,
        dtype: Optional[Any],
        use_flash_attention: bool
    ):
        """Initialize local model from HuggingFace."""
        # Check for required dependencies
        if torch is None:
            raise ImportError(
                "torch is required for local model provider. Install it with: "
                "pip install torch torchvision torchaudio\n"
                "For CUDA support, see: https://pytorch.org/get-started/locally/"
            )
        
        if Qwen3VLForConditionalGeneration is None or AutoProcessor is None:
            raise ImportError(
                "transformers is required for local model provider. Install it with: "
                "pip install git+https://github.com/huggingface/transformers"
            )
        
        if process_vision_info is None:
            raise ImportError(
                "qwen_vl_utils is required for local model provider. "
                "Ensure qwen_vl_utils.py is available in the project directory."
            )
        
        logging.info(f"Loading local Qwen3-VL model: {model_name}")

        if dtype is None:
            dtype = torch.bfloat16

        # Build model kwargs once; load once
        model_kwargs: Dict[str, Any] = dict(
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            # Only for Strix Halo with 96gb set in bios
            #max_memory={0: "90GiB"},
        )

        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logging.info("Flash Attention 2 enabled")
            except Exception:
                logging.warning("flash_attn not installed; using default attention")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        # For MoE Models You need to change to self.model=Qwen3VLMoeForConditionalGeneration.from_pretrained
        logging.info("Local Qwen3-VL agent initialized successfully")
    
    def _init_api_provider(self):
        """Initialize API provider configuration."""
        if not self.api_base_url:
            raise ValueError("api_base_url is required when provider is 'api'")
        
        if not self.api_key:
            raise ValueError("api_key is required when provider is 'api'")
        
        if requests is None:
            raise ImportError("requests library is required for API provider. Install it with: pip install requests")
        
        # Validate API endpoint
        self.api_base_url = self.api_base_url.rstrip('/')
        logging.info(f"Using API provider at: {self.api_base_url}")
        logging.info(f"API Model: {self.model_name}")
        logging.info("API-based Qwen3-VL agent initialized successfully")

    def analyze_screenshot(
        self,
        screenshot_path: str,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Analyze a phone screenshot and determine the next action."""
        try:
            # Load and resize image to prevent OOM
            image = Image.open(screenshot_path)

            # Resize if too large - keep aspect ratio, max dimension 1280
            max_size = 1280
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logging.info(f"Resized image from {Image.open(screenshot_path).size} to {image.size}")

            # Build action history
            history = []
            if context:
                previous_actions = context.get('previous_actions', [])
                for i, act in enumerate(previous_actions[-5:], 1):  # Last 5 actions
                    action_type = act.get('action', 'unknown')
                    element = act.get('elementName', '')
                    history.append(f"Step {i}: {action_type} {element}".strip())

            history_str = "; ".join(history) if history else "No previous actions"

            # Build user query in official format
            user_query = f"""The user query: {user_request}.
Task progress (You have done the following operation on the current device): {history_str}."""

            # Messages in official format
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        {"type": "image", "image": image},
                    ],
                },
            ]

            # Generate response
            action = self._generate_action(messages)

            if action:
                logging.info(f"Generated action: {action.get('action', 'unknown')}")
                logging.debug(f"Full action: {json.dumps(action, indent=2)}")

            return action

        except Exception as e:
            logging.error(f"Error analyzing screenshot: {e}", exc_info=True)
            return None

    def _generate_action(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate an action from the model given messages."""
        if self.provider == "local":
            return self._generate_action_local(messages)
        elif self.provider == "api":
            return self._generate_action_api(messages)
        else:
            logging.error(f"Unknown provider: {self.provider}")
            return None

    def _generate_action_local(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate an action using local model."""
        try:
            # Use processor's chat template
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Collect image/video inputs
            image_inputs, video_inputs = process_vision_info(messages)

            # >>>>>>>>>> IMPORTANT FIX: avoid empty lists (use None) <<<<<<<<<<
            if not image_inputs:
                image_inputs = None
            if not video_inputs:
                video_inputs = None

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,   # None when no videos -> skips video path
                padding=True,
                return_tensors="pt",
            )

            # Move to device
            inputs = inputs.to(self.model.device)

            logging.debug("Generating model response...")

            # Optional: clear cache around generation (works with HIP too)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=self.temperature > 0,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Trim input tokens from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            logging.debug(f"Model output: {output_text}")

            # Parse action
            action = self._parse_action(output_text)
            return action

        except Exception as e:
            logging.error(f"Error generating action with local model: {e}", exc_info=True)
            return None
    
    def _generate_action_api(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Generate an action using OpenAI-compatible API."""
        try:
            # Convert messages to OpenAI format
            api_messages = []
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", [])
                
                # Convert content items
                api_content = []
                for item in content:
                    item_type = item.get("type")
                    
                    if item_type == "text":
                        api_content.append({
                            "type": "text",
                            "text": item.get("text", "")
                        })
                    elif item_type == "image":
                        # Convert PIL image to base64
                        image = item.get("image")
                        if isinstance(image, Image.Image):
                            # Convert to base64
                            buffered = io.BytesIO()
                            image.save(buffered, format="PNG")
                            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            api_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            })
                        elif isinstance(image, str):
                            # Assume it's already a URL or path
                            if image.startswith(("http://", "https://", "data:")):
                                api_content.append({
                                    "type": "image_url",
                                    "image_url": {"url": image}
                                })
                            else:
                                # Load from path and convert - validate path
                                import os
                                from pathlib import Path
                                
                                try:
                                    # Use realpath to resolve symlinks and prevent directory traversal
                                    image_path = Path(image).resolve(strict=True)
                                    
                                    # Additional safety: verify it's actually a file
                                    if not image_path.is_file():
                                        logging.error(f"Path is not a file: {image_path}")
                                        continue
                                    
                                except (OSError, ValueError, RuntimeError) as e:
                                    logging.error(f"Invalid image path: {image} - {e}")
                                    continue
                                
                                img = Image.open(str(image_path))
                                buffered = io.BytesIO()
                                img.save(buffered, format="PNG")
                                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                api_content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_base64}"
                                    }
                                })
                
                api_messages.append({
                    "role": role,
                    "content": api_content
                })
            
            # Make API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": api_messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            endpoint = f"{self.api_base_url}/chat/completions"
            logging.debug(f"Calling API endpoint: {endpoint}")
            
            # SSL certificate verification is enabled by default.
            # For custom SSL certificates, you can pass a path to a CA bundle file:
            # verify='/path/to/certfile' or disable with verify=False (not recommended)
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=60,
                verify=True  # SSL certificate verification enabled
            )
            
            if response.status_code != 200:
                logging.error(f"API request failed: {response.status_code} - {response.text}")
                return None
            
            result = response.json()
            
            # Extract response
            if "choices" not in result or len(result["choices"]) == 0:
                logging.error("No choices in API response")
                return None
            
            output_text = result["choices"][0]["message"]["content"]
            logging.debug(f"API output: {output_text}")
            
            # Parse action
            action = self._parse_action(output_text)
            return action
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error: {e}", exc_info=True)
            return None
        except Exception as e:
            logging.error(f"Error generating action with API: {e}", exc_info=True)
            return None

    def _parse_action(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse action from model output in official format."""
        try:
            # Validate input
            if not text or not isinstance(text, str):
                logging.error("Invalid text input for parsing")
                return None
            
            # Extract tool_call XML content
            match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
            if not match:
                logging.error("No <tool_call> tags found in output")
                logging.debug(f"Output text: {text}")
                return None

            tool_call_json = match.group(1)
            tool_call = json.loads(tool_call_json)

            # Extract arguments
            if 'arguments' not in tool_call:
                logging.error("No 'arguments' in tool_call")
                return None

            args = tool_call['arguments']
            action_type = args.get('action')
            if not action_type:
                logging.error("No 'action' in arguments")
                return None

            # Validate action type
            valid_actions = ['click', 'swipe', 'type', 'wait', 'terminate']
            if action_type not in valid_actions:
                logging.error(f"Invalid action type: {action_type}")
                return None

            # Convert to our internal format
            action: Dict[str, Any] = {'action': action_type}

            # Handle coordinates (convert from 999x999 space to normalized 0-1)
            # Add boundary validation
            if 'coordinate' in args:
                coord = args['coordinate']
                if not isinstance(coord, (list, tuple)) or len(coord) != 2:
                    logging.error(f"Invalid coordinate format: {coord}")
                    return None
                
                # Validate coordinate values are numeric
                try:
                    x_val = float(coord[0])
                    y_val = float(coord[1])
                except (TypeError, ValueError) as e:
                    logging.error(f"Non-numeric coordinates: {coord} - {e}")
                    return None
                
                # Clamp coordinates to valid range
                x = max(0, min(999, x_val))
                y = max(0, min(999, y_val))
                action['coordinates'] = [x / 999.0, y / 999.0]
                
                if x != x_val or y != y_val:
                    logging.warning(f"Coordinates clamped from ({x_val}, {y_val}) to ({x}, {y})")

            if 'coordinate2' in args:
                coord2 = args['coordinate2']
                if not isinstance(coord2, (list, tuple)) or len(coord2) != 2:
                    logging.error(f"Invalid coordinate2 format: {coord2}")
                    return None
                
                # Validate coordinate values are numeric
                try:
                    x2_val = float(coord2[0])
                    y2_val = float(coord2[1])
                except (TypeError, ValueError) as e:
                    logging.error(f"Non-numeric coordinate2: {coord2} - {e}")
                    return None
                
                # Clamp coordinates to valid range
                x2 = max(0, min(999, x2_val))
                y2 = max(0, min(999, y2_val))
                action['coordinate2'] = [x2 / 999.0, y2 / 999.0]
                
                if x2 != x2_val or y2 != y2_val:
                    logging.warning(f"Coordinate2 clamped from ({x2_val}, {y2_val}) to ({x2}, {y2})")

            # Handle swipe - convert to direction for compatibility
            if action_type == 'swipe' and 'coordinates' in action and 'coordinate2' in action:
                start = action['coordinates']
                end = action['coordinate2']
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                
                # Validate meaningful swipe distance
                distance = (dx**2 + dy**2) ** 0.5
                if distance < self.MIN_SWIPE_DISTANCE:
                    logging.warning(f"Swipe distance very small: {distance:.3f} (min: {self.MIN_SWIPE_DISTANCE})")
                
                if abs(dy) > abs(dx):
                    action['direction'] = 'down' if dy > 0 else 'up'
                else:
                    action['direction'] = 'right' if dx > 0 else 'left'

            # Map action names
            if action_type == 'click':
                action['action'] = 'tap'  # our internal name

            # Copy other fields with validation
            if 'text' in args:
                text_input = args['text']
                if not isinstance(text_input, str):
                    logging.error(f"Invalid text type: {type(text_input)}")
                    return None
                action['text'] = text_input
            
            if 'time' in args:
                try:
                    wait_time = float(args['time'])
                    if wait_time < 0:
                        logging.warning(f"Negative wait time: {wait_time}, using 0")
                        wait_time = 0
                    action['waitTime'] = int(wait_time * 1000)  # ms
                except (ValueError, TypeError) as e:
                    logging.error(f"Invalid time value: {args['time']} - {e}")
                    return None
            
            if 'status' in args:
                status = args['status']
                if status not in ['success', 'failure']:
                    logging.warning(f"Unexpected status value: {status}")
                action['status'] = status
                action['message'] = f"Task {status}"

            # Extract thought/action description
            thought_match = re.search(r'Thought:\s*(.+?)(?:\n|$)', text)
            action_match = re.search(r'Action:\s*(.+?)(?:\n|$)', text)
            if thought_match:
                action['reasoning'] = thought_match.group(1).strip().strip('"')
            if action_match:
                action['observation'] = action_match.group(1).strip().strip('"')

            # Validate essentials
            if action['action'] == 'tap' and 'coordinates' not in action:
                logging.error("Tap action missing coordinates")
                return None
            if action['action'] == 'type' and 'text' not in action:
                logging.error("Type action missing text")
                return None

            return action

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON from tool_call: {e}")
            logging.debug(f"Text: {text}")
            return None
        except Exception as e:
            logging.error(f"Error parsing action: {e}", exc_info=True)
            logging.debug(f"Text: {text}")
            return None

    def check_task_completion(
        self,
        screenshot_path: str,
        user_request: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Ask the model if the task has been completed."""
        try:
            # Load and resize image
            image = Image.open(screenshot_path)
            max_size = 1280
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            completion_query = f"""The user query: {user_request}.

You have completed {len(context.get('previous_actions', []))} actions.

Look at the current screen and determine: Has the task been completed successfully?

If complete, use action=terminate with status="success".
If not complete, explain what still needs to be done and use action=terminate with status="failure"."""  # noqa: E501

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": completion_query},
                        {"type": "image", "image": image},
                    ],
                },
            ]

            action = self._generate_action(messages)

            if action and action.get('action') == 'terminate':
                return {
                    "complete": action.get('status') == 'success',
                    "reason": action.get('message', ''),
                    "confidence": 0.9 if action.get('status') == 'success' else 0.7,
                }

            return {"complete": False, "reason": "Unable to verify", "confidence": 0.0}

        except Exception as e:
            logging.error(f"Error checking completion: {e}")
            return {"complete": False, "reason": f"Error: {str(e)}", "confidence": 0.0}
