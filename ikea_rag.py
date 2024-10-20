

#################################
# Download Manuals

import os
import requests

# Define the directory and file URLs
directory = 'ikea_manuals'
files = [
    ("https://www.ikea.com/gb/en/assembly_instructions/fredde-gaming-desk-black__AA-2508156-1-100.pdf", "fredde.pdf"),
    ("https://www.ikea.com/gb/en/assembly_instructions/smagoera-wardrobe-white__AA-2177175-4-100.pdf", "smagoera.pdf"),
    ("https://www.ikea.com/gb/en/assembly_instructions/tuffing-bunk-bed-frame-dark-grey__AA-1627840-10-2.pdf", "tuffing.pdf")
]

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Function to download a file
def download_file(url, filename):
    response = requests.get(url, headers={'User-Agent': 'Mozilla'})
    if response.status_code == 200:
        with open(os.path.join(directory, filename), 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

# Download each file
for url, filename in files:
    download_file(url, filename)


###############################################
# Convert Pdfs To Images


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch
import io
from PIL import Image, ImageDraw
import numpy as np
import csv
import pandas as pd

#from torchvision import transforms

#from transformers import AutoModelForObjectDetection
import torch
import openai
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

OPENAI_API_TOKEN = ""
openai.api_key = OPENAI_API_TOKEN
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN

if not os.path.exists("static"):
    os.makedirs("static")


import fitz
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core.multi_modal_llms.generic_utils import load_image_urls
from llama_index.core import SimpleDirectoryReader


openai_mm_llm = OpenAIMultiModal(
    model="gpt-4-turbo", max_new_tokens=300
)

from llama_index.core.schema import ImageDocument

print(fitz.__version__)

pdfs = ["fredde.pdf", "smagoera.pdf", "tuffing.pdf"]

for pdf_file in pdfs:
    # Split the base name and extension
    output_directory_path, _ = os.path.splitext(pdf_file)

    if not os.path.exists(f"./ikea_manuals/{output_directory_path}"):
        os.makedirs(f"./ikea_manuals/{output_directory_path}")

    # Open the PDF file
    pdf_document = fitz.open(f"./ikea_manuals/{pdf_file}")

    # Iterate through each page and convert to an image
    for page_number in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_number]

        # Convert the page to an image
        pix = page.get_pixmap()

        # Create a Pillow Image object from the pixmap
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Save the image
        image.save(f"./ikea_manuals/{output_directory_path}/{output_directory_path}_page_{page_number + 1}.png")



        image_urls = [
            f"./ikea_manuals/{output_directory_path}/{output_directory_path}_page_{page_number + 1}.png",
        ]

        image_documents = [ImageDocument(image_path=image_path) for image_path in image_urls]

        response = openai_mm_llm.complete(
            prompt=f"Describe the images as an alternative text",
            image_documents=image_documents,
        )

        with open(f"./ikea_manuals/{output_directory_path}/{output_directory_path}_page_{page_number + 1}.txt", "w") as text_file:
            text_file.write(f"{output_directory_path} page {page_number + 1}. "+str(response))

        print(str(response))

    # Close the PDF file
    pdf_document.close()




