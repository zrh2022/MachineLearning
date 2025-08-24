# 使用Hugging Face tokenizer示例
from transformers import GPT2Tokenizer, GPT2Model
import torch


class Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path='C:/Users/18132/.cache/huggingface/hub/models--gpt2/snapshots'
                                          '/607a30d783dfa663caf39e06633721c8d4cfcd7e',
            local_files_only=True)
        # 方法1：GPT-2嵌入矩阵
        model = GPT2Model.from_pretrained(pretrained_model_name_or_path='C:/Users/18132/.cache/huggingface/hub/models'
                                                                        '--gpt2/snapshots'
                                                                        '/607a30d783dfa663caf39e06633721c8d4cfcd7e',
                                          local_files_only=True)
        self.embedding_weights = model.wte.weight  # [vocab_size, d_model]
        # print(f"嵌入矩阵形状: {self.embedding_weights.shape}")  # [50257, 768]
        return

    def get_token_ids(self, text):
        token_ids = self.tokenizer.encode(text)  #
        return token_ids

    def get_tokens(self, text):
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def get_decoded_text(self, token_ids):
        decoded_text = self.tokenizer.decode(token_ids)
        return decoded_text

    # 核心代码：通过索引获取向量
    def get_vectors_by_indices(self, indices):
        """
        embedding_matrix: [vocab_size, d_model] 的嵌入矩阵
        indices: token索引列表
        返回: 对应的向量
        """
        indices_tensor = torch.tensor(indices)
        vectors = self.embedding_weights[indices_tensor]  # 直接索引
        return vectors


def testTokenEncodeAndDecode():
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    text = "Hello world!"
    # encode方法将文本转换为token索引
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")

    # decode方法将索引转换回文本
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded: {decoded_text}")

    # 查看每个token对应的文本
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
