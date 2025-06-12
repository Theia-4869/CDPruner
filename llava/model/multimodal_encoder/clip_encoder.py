import torch
import torch.nn as nn

from transformers import (
    CLIPVisionConfig, CLIPImageProcessor, CLIPVisionModel, CLIPVisionModelWithProjection,
    CLIPTextConfig, CLIPTokenizerFast, CLIPTextModel, CLIPTextModelWithProjection,
)


def hook_k(module, input, output):
    module.k_output = output


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    # CDPruner
    def load_text_tower(self, device_map=None):
        vision_tower_with_projection = CLIPVisionModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.visual_projection = vision_tower_with_projection.visual_projection

        self.text_tokenizer = CLIPTokenizerFast.from_pretrained(self.vision_tower_name)
        self.text_tower = CLIPTextModelWithProjection.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.text_tower.requires_grad_(False)

        self.max_position_embeddings = self.text_tower.config.max_position_embeddings

    def feature_select(self, image_forward_outs, output_attentions=False):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if output_attentions:
            image_attentions = image_forward_outs.attentions[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
            if output_attentions:
                image_attentions = image_attentions[:, :, 0, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
            if output_attentions:
                image_attentions = image_attentions
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        if output_attentions:
            return image_features, image_attentions
        return image_features

    # CDPruner
    @torch.no_grad()
    def forward(self, images, texts=None, output_attentions=False):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # CDPruner
            image_stream = torch.cuda.Stream()
            text_stream = torch.cuda.Stream()
            
            with torch.cuda.stream(image_stream):
                if output_attentions:
                    hook_handle_k = self.vision_tower.vision_model.encoder.layers[self.select_layer].self_attn.k_proj.register_forward_hook(hook_k)
                image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype),
                                                    output_hidden_states=True,
                                                    output_attentions=output_attentions)
                image_outputs = self.feature_select(image_forward_outs, output_attentions=output_attentions)
                if output_attentions:
                    image_features = (
                        image_outputs[0].to(images.dtype),
                        image_outputs[1].to(images.dtype),
                        self.vision_tower.vision_model.encoder.layers[self.select_layer].self_attn.k_proj.k_output[:, 1:].to(images.dtype),
                        image_forward_outs.hidden_states[self.select_layer][:, :1].to(images.dtype),
                    )
                    hook_handle_k.remove()
                else:
                    image_features = image_outputs.to(images.dtype)
            
            if texts is not None:
                with torch.cuda.stream(text_stream):
                    text_inputs = self.text_tokenizer(text=texts, return_tensors="pt")
                    text_segment = (text_inputs.input_ids.shape[1] - 1) // self.max_position_embeddings + 1
                    text_padding = self.max_position_embeddings * text_segment - text_inputs.input_ids.shape[1]
                    text_inputs = {
                        k: torch.cat([v, v.new_zeros((v.shape[0], text_padding))], 
                                     dim=1).reshape(-1, self.max_position_embeddings).to(device=self.device)
                        for k, v in text_inputs.items()
                    }
                    text_embeds = self.text_tower(**text_inputs).text_embeds
            
            torch.cuda.synchronize()

            if texts is not None:
                image_embeds = self.vision_tower.vision_model.post_layernorm(image_outputs)
                image_embeds = self.vision_tower.visual_projection(image_embeds.float())
                image_features = (image_features, image_embeds, text_embeds)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
