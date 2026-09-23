import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()


client = InferenceClient(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

prompt = """
A cute young couple dancing 
"""

image = client.text_to_image(
    prompt=prompt,
    model="lvladikov/Krea2-Turbo-Distill-4step-LoRA"
)

image.save("generated_image.png")

print("Image generated successfully!")