from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = AutoImageProcessor.from_pretrained('/kaggle/input/models/metaresearch/dinov2/pytorch/small/1')
model = AutoModel.from_pretrained('/kaggle/input/models/metaresearch/dinov2/pytorch/small/1')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state