# 多模态LLM

模态：每种数据类型称为一种模态

多模态模型：能够处理文本、图形等多种数据类型的模型

&nbsp;

## 视觉Transformer

视觉Transformer（Vision Transformer, ViT），核心功能是将非结构化的图像数据转换为可用于分类等任务的数值表示。

图形编码：将原始图形切割成规则排列的图像块，进而实现特征提取。

图像被切割成图像块，展平输入再实施线性嵌入操作，将其转换为数值化的嵌入向量作为Transformer模型的标准化输入。

&nbsp;

## 多模态嵌入模型

主流方案：对比语言-图像预训练模型（Contrastive Language-Image Pre-Training, CLIP），能同时计算图像嵌入和文本嵌入，应用场景支持零样本分类，语义聚类，跨模态检索，生成引导。

训练方法：对比学习（contrastive learning）

使用包含图像和其对应描述文本的数据集作为训练集。第一步为每一组图像和描述文本创建两个向量表示，CLIP采用双编码器架构，文本编码器处理描述文本，生成语义嵌入，图像编码器提取视觉特征，生成图像嵌入；第二步使用余弦相似度计算文本嵌入和图像嵌入之间的匹配程度，最大化匹配图文对的相似度，同时最小化非匹配对的相似度；第三步根据预期相似度更新文本编码器和图像编码器参数。训练过程中还需引入无关图形和文本描述作为负例。

&nbsp;

## OpenCLIP

使用模型：openai/clip-vit-base-patch32

下载模型：modelscope download --model openai-mirror/clip-vit-base-patch32 --local_dir E:\huggingface\models\openai-clip-vit-base-patch32

```
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel
from langchain_core.prompts import PromptTemplate
from PIL import Image


puppy_path = "E:/huggingface/images/puppy.png"
image = Image.open(puppy_path).convert("RGB")
caption = "a puppy playing in the snow"

model_path = "E:/huggingface/models/openai-clip-vit-base-patch32"

# Load a tokenizer to preprocess the text
clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_path)

# Load a processor to preprocess the images
clip_processor = CLIPProcessor.from_pretrained(model_path)

# Main model for generating text and image embeddings
model = CLIPModel.from_pretrained(model_path)

# Tokenize our input
inputs = clip_tokenizer(caption, return_tensors="pt")

# Create a text embedding
text_embedding = model.get_text_features(**inputs).pooler_output

# Preprocess image
processed_image = clip_processor(
    text=None, images=image, return_tensors='pt'
)['pixel_values']

# Create the image embedding
image_embedding = model.get_image_features(pixel_values=processed_image).pooler_output

# Normalize the embeddings
text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

# Calculate their similarity
text_embedding = text_embedding.detach().cpu().numpy()
image_embedding = image_embedding.detach().cpu().numpy()
score = text_embedding @ image_embedding.T
print(score)
```

```
Loading weights: 100%|██████████| 398/398 [00:00<00:00, 17267.65it/s]
[[0.331469]]
```

&nbsp;

## 让文本生成模型具有多模态能力

BLIP-2

通过构建名为查询式Transformer（Querying Transformer, Q-Former）的智能桥梁，巧妙连接预训练视觉编码器与预训练LLM，而非重新构建整个系统架构。仅需训练Q-former模块，无需从零训练视觉编码器和LLM。

&nbsp;

图像描述

模型：Salesforce/blip2-opt-2.7b

下载模型：modelscope download --model Salesforce/blip2-opt-2.7b --local_dir E:\huggingface\models\Salesforce_blip2-opt-2.7b

```
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image


model_path = "E:/huggingface/models/Salesforce_blip2-opt-2.7b"
# Load processor and main model
blip_processor = AutoProcessor.from_pretrained(
    model_path,
    revision="51572668da0eb669e01a189dc22abe6088589a24"  # Choose specific model because of: https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/39
)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_path,
    revision="51572668da0eb669e01a189dc22abe6088589a24",
    torch_dtype=torch.float16
)

# Send the model to GPU to speed up inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load image of a supercar
car_path = "E:/huggingface/images/car.png"
image = Image.open(car_path).convert("RGB")

# Convert an image into inputs and preprocess it
inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)

# Generate image ids to be passed to the decoder (LLM)
generated_ids = model.generate(**inputs, max_new_tokens=20)

# Generate text from the image ids
generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
generated_text = generated_text[0].strip()
print(generated_text)
```

```
Loading weights: 100%|██████████| 1247/1247 [00:08<00:00, 141.12it/s]
an orange supercar driving on the road at sunset
```

&nbsp;

基于聊天的多模态提示词

```
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image


model_path = "E:/huggingface/models/Salesforce_blip2-opt-2.7b"
# Load processor and main model
blip_processor = AutoProcessor.from_pretrained(
    model_path,
    revision="51572668da0eb669e01a189dc22abe6088589a24"  # Choose specific model because of: https://huggingface.co/Salesforce/blip2-opt-2.7b/discussions/39
)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_path,
    revision="51572668da0eb669e01a189dc22abe6088589a24",
    torch_dtype=torch.float16
)

# Send the model to GPU to speed up inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load image of a supercar
car_path = "E:/huggingface/images/car.png"
image = Image.open(car_path).convert("RGB")

# Visual Question Answering
prompt = "Question: Write down what you see in this picture. Answer:"

# Process both the image and the prompt
inputs = blip_processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

# Generate text
generated_ids = model.generate(**inputs, max_new_tokens=30)
generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
generated_text = generated_text[0].strip()
print(generated_text)

# Chat-like prompting
prompt = "Question: Write down what you see in this picture. Answer: A sports car driving on the road at sunset. Question: What would it cost me to drive that car? Answer:"

# Generate output
inputs = blip_processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=30)
generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
generated_text = generated_text[0].strip()
print(generated_text)
```

```
Loading weights: 100%|██████████| 1247/1247 [00:04<00:00, 285.55it/s]
Question: Write down what you see in this picture. Answer: A sports car driving on the road at sunset
Question: Write down what you see in this picture. Answer: A sports car driving on the road at sunset. Question: What would it cost me to drive that car? Answer: $1,000,000
```
