## Demo Script Usage

The `demo.py` script provides an easy way to interact with MetaMorph for both text and image generation. It supports both image and video inputs and can operate in two modes: standard chat and vision-enhanced chat.

### Basic Usage

```bash
# Basic chat without vision (will not process the image)
python demo.py --input-path path/to/image.jpg --prompt "Generate an image of a cat"

# Chat with vision capabilities (will analyze the image)
python demo.py --input-path path/to/image.jpg --prompt "What is in this image?" --chat-with-vision

# Process a video with vision
python demo.py --input-path path/to/video.mp4 --prompt "Describe what's happening in this video" --chat-with-vision
```

### Key Features

- **Two Chat Modes**: 
  - Standard mode: Regular text generation
  - Vision mode: Enables visual understanding with `--chat-with-vision` flag
- **Image/Video Support**: Automatically handles both image files (.jpg, .png) and video files (.mp4, .avi, .mov)
- **Image Generation**: Automatically generates visualizations with multiple guidance scales (6.0, 7.5, 10.0, 12.5)
- **Output Management**: Saves generated images to specified output directory (default: "outputs")

### Arguments

- `--input-path`: Path to input image or video (default: "cat.jpg")
- `--prompt`: Text prompt for the model (default: "What is this animal?")
- `--output-dir`: Directory to save generated images (default: "outputs")
- `--chat-with-vision`: Flag to enable vision-language chat mode (required for image analysis)

### Example Outputs

The script generates multiple variations of visualizations for each input, using different guidance scales. Output files are named in the format: `output_<index>_scale_<guidance_scale>.png`

## Operational Modes

MetaMorph can operate in several modes, each designed for specific use cases:

### 1. Image Generation Mode
Generate images from text descriptions, leveraging advanced language understanding and reasoning capabilities.

```bash
# Basic image generation
python demo.py --prompt "Generate an image of a cat"


# Leverage LLM knowledge
python demo.py --prompt "Generate an image of Chhogori"

# Visual puzzles and riddles
python demo.py --prompt "Generate an image of: An animal, this large mammal shares its name with a constellation often visible in the night sky and associated with northern part of the world"

```


The model can understand and solve puzzles. For example, in the constellation riddle, it would recognize "Ursa Major" (The Great Bear) as the answer and generate the appropriate image.


### 2. Vision Chat Mode
Interactive conversations about images or videos with visual understanding.
```bash
# Use --chat-with-vision flag
python demo.py --input-path image.jpg --prompt "What emotions do you sense in this image?" --chat-with-vision
python demo.py --input-path video.mp4 --prompt "Describe the main events in this clip" --chat-with-vision
```

### 3. Visual Thinking Mode
Enables the model to use visual tokens in its reasoning process when beneficial.
```bash
# Don't use --chat-with-vision flag
python demo.py --prompt "Let's think about this visually: What is the type of the hat?" --input-path image.jpg --chat-with-vision
```

### 4. Image Transformation Mode
Modify or transform input images based on textual instructions.
```bash
# Use --chat-with-vision flag
python demo.py --input-path photo.jpg --prompt "Make it a cartoon" --chat-with-vision
```

### Mode Selection Guidelines

- **Input Images**: Required for Vision Chat and Image Transformation modes
- **--chat-with-vision flag**: 
  - Use with: Vision Chat, Image Transformation
  - Don't use with: Image Generation
- **Prompt Structure**:
  - Image Generation: "Generate an image of {description}"
  - Visual Thinking: "Let's think about this visually: {prompt}"
  - Vision Chat: Any natural question or instruction
  - Image Transformation: Direct modification commands

### Output Behavior

- **Image Generation**: Creates multiple variations with different guidance scales
- **Vision Chat**: Provides textual analysis and understanding
- **Visual Thinking**: May generate intermediate visual representations during reasoning
- **Image Transformation**: Produces modified versions of the input image