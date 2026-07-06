import torch
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
model_name="openai/clip-vit-base-patch32"
model=CLIPModel.from_pretrained(model_name)
processor=CLIPProcessor.from_pretrained(model_name)
img_url="https://images.unsplash.com/photo-1478098711619-5ab0b478d6e6"
image=Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
labels=[
    "a cat",
    "a dog",
    "a horse",
    "a car"
]
inputs=processor(
    text=labels,
    images=image,
    return_tensors="pt",
    padding=True
)
with torch.no_grad():
  outputs=model(**inputs)
  logits=outputs.logits_per_image
  probs=logits.softmax(dim=1)
  print("Predictions:")
  for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob.item():.4f}")
  best_idx=probs.argmax().item()
  print(f"\nPredicted class: {labels[best_idx]}")
