import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int):   # h = number of heads
        super.__init__(MultiHeadAttention, self)
        self.d_model = d_model
        self.h = h

        self.d_k, self.d_v = d_model / h

        self.query_weights_multiplication = nn.Linear(d_model, d_model)  # In paper it is d_model, d_k but for efficient integration I chose d_model, d_model then split the heads
        self.key_weights_multiplication = nn.Linear(d_model, d_model)
        self.value_weights_multiplication = nn.Linear(d_model, d_model)

    def _separate_head(self, vec):

        batch_size, sequence_length, d_model = vec.size()
        # vec = (batch_size, sequence_length, h, d_k)
        vec = vec.view(batch_size, sequence_length, self.h, d_model)
        # vec = (batch_size, h, sequence_length , d_k) by transposing index 1 and 2
        return vec.transpose(1, 2)

    def _dot_product_attention(self, query, key, value, mask = None):
        # (batch_size, h, sequence_length , d_k)
        QK = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5) #(batch_size, h, sequence_length, sequence_length)
        # masking out (setting to -inf) all values in the input of the softmax which correspond to illegal connetction
        if mask:
            QK = QK.masked_fill(mask ==0, -1e9)

        softmax_QK = torch.softmax(QK, dim=-1)
        #(batch_size, h, sequence_length, sequence_length) * (batch_size, h, sequence_length, d_model)
        attention = torch.matmul(softmax_QK, value)
        #(batch_size, h, sequence_length, d_model)
        return attention

    def forward(self, query, key, value, mask = None):
        # query = (batch_size, seqence_length, d_model) 
        # After multiplication for adding learnable parameters (batch_size, sequence_length, d_model) * (d_model, d_model) = (batch_size, sequence_length, d_model)
        query = self.query_weights_multiplication(query)
        key = self.key_weights_multiplication(key)
        value = self.value_weights_multiplication(value)
        # must be  query = (batch_size, h, sequence_length , d_k)
        query = self._separate_head(query)
        key = self._separate_head(key)
        value = self._separate_head(value)

        # output must be (batch_size, sequence_length, d_model)




