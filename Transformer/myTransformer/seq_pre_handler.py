import math

import torch

from tokenizer import Tokenizer
from positional_encoding import PositionalEncoding


class TextPreHandler:
    def __init__(self, seq_max_tokens, d_model):
        self.token = Tokenizer()
        self.seq_max_tokens = seq_max_tokens
        self.d_model = d_model

    def pre_handle_texts(self, texts):
        # 将句子转到成token_ids
        tokenizer = Tokenizer()
        # tokens_list = [tokenizer.get_tokens(text) for text in texts]  # shape：[6, ?]
        # print(tokens_list)
        token_ids_list = [tokenizer.get_token_ids(text) for text in texts]  # shape：[6, ?]
        # print(token_ids_list)

        # 2. Padding
        padded_token_ids_list = []
        for ids in token_ids_list:
            if len(ids) > self.seq_max_tokens:  # 如果超过了最多的词元数目
                padded = ids[:self.seq_max_tokens]  # 截断
            else:
                padded = ids + [0] * (self.seq_max_tokens - len(ids))  # 填充
            padded_token_ids_list.append(padded)
        # print(padded_token_ids_list)  # shape：[6, 10]

        # shape：[6, 10, 748],但是实际为列表，列表里每个元素为[10, 768]的tensor，共计6个
        token_vectors_list = [tokenizer.get_vectors_by_indices(ids) for ids in padded_token_ids_list]
        # print(token_vectors_list)

        # 使用stack，会在指定维度堆叠
        token_tensor = torch.stack(token_vectors_list, dim=0) * math.sqrt(self.d_model)
        # 实际的输入向量
        # print("转到到tensor：" + str(token_tensor.shape))  # shape：[6, 10, 768]
        return token_tensor

    def pre_handle_texts_v2(self, texts):
        # 将句子转到成token_ids
        tokenizer = Tokenizer()
        # tokens_list = [tokenizer.get_tokens(text) for text in texts]  # shape：[6, ?]
        # print(tokens_list)
        token_ids_list = [tokenizer.get_token_ids(text) for text in texts]  # shape：[6, ?]
        # print(token_ids_list)

        # 2. Padding
        padded_token_ids_list = []
        for ids in token_ids_list:
            if len(ids) > self.seq_max_tokens:  # 如果超过了最多的词元数目
                padded = ids[:self.seq_max_tokens]  # 截断
            else:
                padded = ids + [0] * (self.seq_max_tokens - len(ids))  # 填充
            padded_token_ids_list.append(padded)
        # print(padded_token_ids_list)  # shape：[6, 10]

        padded_token_ids_list = torch.tensor(padded_token_ids_list);
        return padded_token_ids_list

    def texts_post_handle(self, texts):
        ori_texts = [self.token.get_decoded_text(text) for text in texts]
        return ori_texts