from PIL import Image as im
import numpy as np
from pytriton.client import ModelClient


def generate_image(prompt, img_file_name):
    prompts = np.array([prompt.encode()])  # Triton can handle multiple prompts at a time

    with ModelClient("localhost", "text_to_image") as client:
        print("Sending to Triton server, please wait a few seconds...")
        result_dict = client.infer_sample(prompts)
        img = im.fromarray(result_dict["image"])
        img.save(img_file_name)
        print("Images saved as", img_file_name)

while True:
    prompt = input("What image would you like created (press q to quit): ")
    if prompt.lower() == "q":
        break

    file_name = input("What filename would you like to save this image as: ")
    generate_image(prompt, file_name)
    print("------------------------")

