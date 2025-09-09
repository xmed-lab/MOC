import torch
from timm.models import create_model
from .musk import utils, modeling
from transformers import XLMRobertaTokenizer
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torchvision.transforms as transforms


def musk_model(checkpoint_path="/home/txiang/pathology/CLAM/models/musk.safetensors"):
    model_config = "musk_large_patch16_384"
    model = create_model(model_config, checkpoint_path=checkpoint_path)
    return model

def get_musk_tokenizer(tokenizer_path="/home/txiang/pathology/CLAM/models/musk/models/tokenizer.spm"):
    tokenizer = XLMRobertaTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

def encode_img_musk(model, img_tensor):
    return model(image=img_tensor, with_head=True, out_norm=True, ms_aug=False)[0]

# def encode_txt_musk(model, sentence, tokenizer=None, device=None):
#     if device is None:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     if tokenizer is None:
#         tokenizer = get_musk_tokenizer()
#     text_tokens, padding_mask = utils.xlm_tokenizer(sentence, tokenizer)
#     text_tokens = torch.tensor(text_tokens).unsqueeze(0)

#     text_embeddings = model(text_description=text_tokens.to(device), 
#                             padding_mask=padding_mask.to(device), 
#                             with_head=True, 
#                             out_norm=True, 
#                             ms_aug=False)[1]
#     return text_embeddings

'''
for image-text zero shot
model(image=img, with_head=True, out_norm=True, ms_aug=False)
image: Input image tensor.
            text_description: Input text tokens.
            padding_mask: Padding mask for text.
            return_global: Whether to return global CLS token.
            with_head: Whether to apply linear heads.
            out_norm: Whether to normalize output embeddings.
            ms_aug: Enable multiscale feature augmentation. 
            scales: List of scales for multiscale feature augmentation.
            max_split_size: Maximum split size for multiscale forward.

# >>>>>>>>>>>> process image >>>>>>>>>>> #
# load an image and process it
img_size = 384
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size, interpolation=3, antialias=True),
    torchvision.transforms.CenterCrop((img_size, img_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
])

img = Image.open('assets/lungaca1014.jpeg').convert("RGB")  # input image
img_tensor = transform(img).unsqueeze(0)
with torch.inference_mode():
    image_embeddings = model(
        image=img_tensor.to(device, dtype=torch.float16),
        with_head=True,  # We only use the retrieval head for image-text retrieval tasks.
        out_norm=True
        )[0]  # return (vision_cls, text_cls)
# <<<<<<<<<<< process image <<<<<<<<<<< #

# >>>>>>>>>>> process language >>>>>>>>> #
# load tokenzier for language input
tokenizer = XLMRobertaTokenizer("./musk/models/tokenizer.spm")
# labels = ["lung adenocarcinoma",
#             "benign lung tissue",
#             "lung squamous cell carcinoma"]
labels = ["lung adenocarcinoma"]

texts = ['histopathology image of ' + item for item in labels]
text_ids = []
paddings = []
for txt in texts:
    txt_ids, pad = utils.xlm_tokenizer(txt, tokenizer, max_len=100)
    text_ids.append(torch.tensor(txt_ids).unsqueeze(0))
    paddings.append(torch.tensor(pad).unsqueeze(0))

text_ids = torch.cat(text_ids)
paddings = torch.cat(paddings)
with torch.inference_mode():
    text_embeddings = model(
        text_description=text_ids.to(device),
        padding_mask=paddings.to(device),
        with_head=True, 
        out_norm=True
    )[1]  # return (vision_cls, text_cls)
# <<<<<<<<<<<< process language <<<<<<<<<<< #

'''

def get_musk_transforms():
    img_size = 384
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=3, antialias=True),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ])
    return transform

# model = musk_model()
# print(model)
# from PIL import Image
# tokenizer = get_musk_tokenizer()
# transform = get_musk_transforms()
# img_tensor = torch.rand(1, 3, 384, 384)
# # img = Image.fromarray(img.numpy())

# # img_tensor = transform(img)
# img_embedding = encode_img_musk(model, img_tensor)

# sentence = "A cat is sitting on the window"
# text_embedding = encode_txt_musk(model, sentence, tokenizer)
# print(img_embedding.shape)
# print(text_embedding.shape)
