import torch
from diffusers import StableDiffusionPipeline
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from pytriton.decorators import batch
import numpy as np

PIPELINE = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, revision="fp16").to("cuda") # using .to("cuda") enables GPU usage

def convert_to_numpy(img):
    return np.asarray(img)

# this function will generate a single image but could be changed to generate multiple
def _generate_image(inputs):
    prompt = inputs[0]["prompt"]
    prompts = np.squeeze(np.char.decode(prompt.astype("bytes"), "utf-8")).tolist()
    imgs = PIPELINE(prompts, height=512, width=768).images

    return [{"image": np.asarray(list(map(convert_to_numpy, imgs)))}]


triton = Triton()
triton.bind(
    model_name="text_to_image",
    infer_func=_generate_image,
    inputs=[
        Tensor(name="prompt", dtype=np.bytes_, shape=(-1,))
    ],
    outputs=[
        Tensor(name="image", dtype=np.uint8, shape=(-1,-1,-1)),
    ],
    config=ModelConfig(max_batch_size=8)
)

triton.run()
