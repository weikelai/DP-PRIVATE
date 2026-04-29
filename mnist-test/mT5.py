import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 下载 nltk 需要的资源
nltk.download('punkt')

# 加载更大的 mT5-base 模型和 tokenizer
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# 英文原句
english_sentences = [
    "In the heart of the bustling city, where skyscrapers reach for the clouds and the streets are always alive with activity, lies a quiet park, an oasis of calm amidst the chaos.",
    "The sun sets over the horizon, painting the sky in hues of orange and pink, casting long shadows across the landscape.",
    "Li Ming, male, from California, USA, enjoys playing tennis. He usually goes to a nearby bar every Friday to have a glass of whisky. He loves his daughter and wife very much. His wife works as a nurse in the dental department of the hospital.",
    "Wang Ming, female, from Beijing, China, enjoys music and often takes walks in the park on weekends. She loves tasting various cuisines, especially seafood. Her husband is a psychologist working at a private hospital."
]

# 生成 mT5-base 翻译结果
translated_sentences = []
for sentence in english_sentences:
    input_text = "<zh> " + sentence  # 指定目标语言
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=100)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    translated_sentences.append(translated_text)

# 计算 BLEU 分数
smoothie = SmoothingFunction().method1
bleu_scores = []

# 为每个句子计算 BLEU 分数
for ref, trans in zip(english_sentences, translated_sentences):
    reference_tokens = [nltk.word_tokenize(ref)]  # 参考译文（英语）
    translation_tokens = nltk.word_tokenize(trans)  # mT5 生成的翻译（中文）
    score = sentence_bleu(reference_tokens, translation_tokens, smoothing_function=smoothie)
    bleu_scores.append(score)

# 输出翻译和 BLEU 分数
for i, (ref, trans, score) in enumerate(zip(english_sentences, translated_sentences, bleu_scores), 1):
    print(f"测试句 {i}:")
    print(f"原始句子: {ref}")
    print(f"mT5-base 翻译结果: {trans}")
    print(f"BLEU 分数: {score:.4f}\n")
