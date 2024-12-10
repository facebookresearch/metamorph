import argparse
import torch
from PIL import Image
import os
from metamorph.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from metamorph.conversation import conv_templates
from metamorph.mm_utils import process_images, tokenizer_image_token
from load_metamorph import load_metamorph_model
from load_visualization import load_visualization
from decord import VideoReader, cpu
import re

def load_image(image_path):
    """Load image from path"""
    return Image.open(image_path).convert("RGB")

def process_video_to_images(video_path, fps=1.0):
    """Extract frames from video"""
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    avg_fps = vr.get_avg_fps()
    total_frames = len(vr)
    frame_interval = int(avg_fps / fps)
    
    images = []
    for i in range(0, total_frames, frame_interval):
        frame = vr[i].asnumpy()
        frame_rgb = frame[:, :, ::-1]
        images.append(Image.fromarray(frame_rgb))
    
    print(f"Extracted {len(images)} frames from video")
    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, default="cat.jpg", 
                      help="Path to input image or video file")
    parser.add_argument("--prompt", type=str, default="What is this animal?",
                      help="Prompt for the model")
    parser.add_argument("--output-dir", type=str, default="outputs",
                      help="Directory to save outputs")
    parser.add_argument("--chat-with-vision", action="store_true",
                      help="Whether to generate visualizations")
    args = parser.parse_args()

    # Create output directory if we're generating images
    os.makedirs(args.output_dir, exist_ok=True)

    # Load MetaMorph model
    print("Loading MetaMorph model...")
    tokenizer, model, image_processor, context_len = load_metamorph_model()
    
    # Load visualization models only if needed
    pipeline, projector = None, None

    print("Loading visualization models...")
    pipeline, projector = load_visualization()

    # Load input (image or video)
    if args.input_path.endswith(('.mp4', '.avi', '.mov')):
        images = process_video_to_images(args.input_path)
    else:
        images = [load_image(args.input_path)]

    # Process images
    image_sizes = [img.size for img in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    num_frames = len(images)

    # Prepare prompt
    qs = args.prompt
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    
    if args.chat_with_vision:
        if IMAGE_PLACEHOLDER in qs:
            if model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se * num_frames, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN * num_frames, qs)
        else:
            if model.config.mm_use_im_start_end:
                qs = image_token_se * num_frames + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN * num_frames + "\n" + qs

    # Prepare conversation
    conv = conv_templates["llama3"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Prepare input tokens
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).cuda()

    # Generate response
    print("Generating response...")
    with torch.inference_mode():
        output_ids, image_embeds = model.generate(
            input_ids,
            images=images_tensor if args.chat_with_vision else None,
            image_sizes=image_sizes,
            output_image=True, 
            do_sample=True,
            temperature=0.0,
            use_customize_greedy=True,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )
    
    # Process text output
    text_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print("\nModel Response:")
    print(text_output)

    # Only process image embeddings if we want to generate images
    if image_embeds is not None:
        # Process image embeddings for visualization
        image_embeds = image_embeds.reshape(1, *image_embeds.shape)
        num_tokens = image_embeds.shape[1]
        num_images = num_tokens // 64

        # Generate visualizations for each image embedding
        guidance_scales = [6.0, 7.5, 10.0, 12.5]
        
        for img_idx in range(num_images):
            print(f"\nProcessing visualization {img_idx + 1}/{num_images}")
            
            # Extract embeddings for current image
            current_embeds = image_embeds[:, img_idx * 64:(img_idx + 1) * 64, :]
            
            # Generate image with different guidance scales
            for scale in guidance_scales:
                # Project embeddings
                with torch.no_grad():
                    projected_embed = projector(current_embeds)
                    
                    # Pad from 64 to 77 tokens if needed
                    if projected_embed.shape[1] != 77:
                        padding = torch.zeros(
                            1, 77 - projected_embed.shape[1], projected_embed.shape[2],
                            device=projected_embed.device
                        )
                        projected_embed = torch.cat([projected_embed, padding], dim=1)
                    
                    # Generate image
                    output = pipeline(
                        prompt_embeds=projected_embed,
                        negative_prompt_embeds=torch.zeros_like(projected_embed),
                        guidance_scale=scale,
                        num_inference_steps=50,
                    ).images[0]
                    
                    # Save output
                    output_path = os.path.join(
                        args.output_dir, 
                        f"output_{img_idx+1}_scale_{scale}.png"
                    )
                    output.save(output_path)
                    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    main()