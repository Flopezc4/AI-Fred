from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import time
import os


processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')

def img_to_txt(images):
    txt = []
    for img in images:
        tensor_pixel = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(tensor_pixel)
        txt.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
    return txt


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename)).convert("RGB")
        if img is not None:
            images.append(img)
    return images


dir = '/home/rodrigo/Workspace/IA_2022/trocr/images/output2'

images = load_images_from_folder(folder=dir)

plates = img_to_txt(images=images)

for plate in plates:
    print(plate)


