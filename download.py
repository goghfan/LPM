from transformers import BertModel
BertModel.from_pretrained("bert-base-uncased", cache_dir="./bert_local")
# 这会把模型下载到 ./bert_local 目录下 (或者你指定的其他缓存目录)