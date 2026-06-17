import math

import node_helpers
import comfy.utils
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io


# Edit system prompt from the reference pipeline (pipeline_boogu.py: SYSTEM_PROMPT_4_TI2I /
# SYSTEM_PROMPT_4_I2I, both == SYSTEM_PROMPT_4_TI2I_UNIFIED).
BOOGU_EDIT_SYSTEM = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
BOOGU_EDIT_TEMPLATE = "<|im_start|>system\n" + BOOGU_EDIT_SYSTEM + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


class TextEncodeBooguEdit(io.ComfyNode):
    """Boogu-Image Edit conditioning

    Qwen3-VL vision tokens (instructionunderstanding)
    VAE reference latent (image identity).
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeBooguEdit",
            category="model/conditioning/boogu",
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input("image1", optional=True),
                io.Image.Input("image2", optional=True),
                io.Image.Input("image3", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae=None, image1=None, image2=None, image3=None) -> io.NodeOutput:
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        image_prompt = ""

        for image in images:
            if image is None:
                continue
            samples = image.movedim(-1, 1)

            # Vision tower input: the reference caps the VLM image at 384x384
            # (max_vlm_input_pil_pixels in pipeline_boogu.py) -> ~147 vision tokens.
            total = int(384 * 384)
            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1)[:, :, :, :3])
            # Images go before the instruction text
            image_prompt += VISION_BLOCK

            # Reference latent: align to 16 px (VAE /8 * patch_size 2).
            if vae is not None:
                total = int(1024 * 1024)
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by / 16.0) * 16
                height = round(samples.shape[2] * scale_by / 16.0) * 16
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

        # positive: instruction + vision tokens (+ ref latent)
        pos_tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=BOOGU_EDIT_TEMPLATE)
        positive = clip.encode_from_tokens_scheduled(pos_tokens)

        # negative: empty text, no vision tokens (+ ref latent)
        neg_tokens = clip.tokenize("", images=[], llama_template=BOOGU_EDIT_TEMPLATE)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        return io.NodeOutput(positive, negative)


class BooguExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TextEncodeBooguEdit,
        ]


async def comfy_entrypoint() -> BooguExtension:
    return BooguExtension()
