import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def visualize(args):
    model = CLIPModel.from_pretrained(args.model_path)
    processor = CLIPProcessor.from_pretrained(args.model_path)

    template = "Is there a {} in the image?"
    object_dict = {
        "COCO_val2014_000000017708.jpg": ["bird", "boat", "rock", "lake"],
        "COCO_val2014_000000217397.jpg": ["phone", "plate", "bottle", "glass"],
        "COCO_val2014_000000480122.jpg": ["banana", "bowl", "table", "kettle"],
        "COCO_val2014_000000536073.jpg": ["blender", "tequila", "lime", "knife"],
    }

    image_mean = processor.image_processor.image_mean
    image_std = processor.image_processor.image_std

    for image_name in os.listdir(args.image_folder):
        output_dir = os.path.join(args.output_folder, image_name.split(".")[0])
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(os.path.join(args.image_folder, image_name))
        vision_inputs = processor(
            images=image,
            return_tensors="pt",
            size={
                "height": 336,
                "width": 336,
            },
        ) # pixel_values

        resized_image = vision_inputs.pixel_values[0].permute(1, 2, 0)
        resized_image = resized_image * torch.tensor(image_std) + torch.tensor(image_mean)
        resized_image = (resized_image.numpy() * 255).astype(np.uint8)
        resized_image = Image.fromarray(resized_image)
        resized_image = resized_image.resize((336, 336))
        original_image = np.array(resized_image)

        for obj in object_dict[image_name]:
            vision_outputs = model.vision_model(**vision_inputs, output_hidden_states=True)

            vision_embeds = vision_outputs.hidden_states[-2][0, 1:]
            vision_embeds = model.vision_model.post_layernorm(vision_embeds)
            vision_embeds = model.visual_projection(vision_embeds)

            text = template.format(obj)
            text_inputs = processor(
                text=text,
                return_tensors="pt",
                padding=True,
            ) # input_ids, attention_mask

            text_outputs = model.text_model(**text_inputs)
            text_embeds = text_outputs.pooler_output
            text_embeds = model.text_projection(text_embeds)

            text_embeds_norm = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
            vision_embeds_norm = vision_embeds / vision_embeds.norm(p=2, dim=-1, keepdim=True)
            sim_map = torch.matmul(text_embeds_norm, vision_embeds_norm.t())

            sim_map = -sim_map[0].reshape(24, 24).unsqueeze(0).unsqueeze(0)
            sim_map = F.interpolate(sim_map, size=(336, 336), mode='nearest')
            sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
            sim_map = np.uint8(sim_map.squeeze(0).squeeze(0).detach() * 255)

            jet_colormap = plt.get_cmap('jet')
            sim_map_colored = jet_colormap(sim_map)
            sim_map_colored = np.uint8(sim_map_colored * 255)

            alpha = 0.5
            overlay = np.uint8(original_image * (1 - alpha) + sim_map_colored[:, :, :3] * alpha)
            plt.imsave(os.path.join(output_dir, f"{obj}.png"), overlay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--image-folder", type=str, default="playground/data/samples")
    parser.add_argument("--output-folder", type=str, default="playground/data/visualize")
    args = parser.parse_args()

    visualize(args)
