from torch import nn
import torch
import math
import numpy as np

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        # ipdb.set_trace()
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_attention_heads=4,
                 attention_probs_dropout_prob=0.2, hidden_dropout_prob=0.2):
        super(SelfAttention, self).__init__()

        self.attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.AE = nn.Linear(input_size, hidden_size)

        self.query_head = nn.ModuleList()
        self.key_head = nn.ModuleList()
        self.value_head = nn.ModuleList()
        for i in range(num_attention_heads):
            self.query_head.append(nn.Linear(hidden_size, hidden_size))
            self.key_head.append(nn.Linear(hidden_size, hidden_size))
            self.value_head.append(nn.Linear(hidden_size, hidden_size))

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor, return_attention=False, is_drop=False):

        # [cell, emb] <- [cell, genes]
        input_tensor = self.AE(input_tensor)
        #
        outputs = []
        for i in range(self.attention_heads):
            query = self.query_head[i]
            key = self.key_head[i]
            value = self.value_head[i]

            query_layer = query(input_tensor)
            key_layer = key(input_tensor)
            value_layer = value(input_tensor)

            # [cells, cells] = [cell, emb]*[emb, cell]
            attention_scores = torch.matmul(query_layer, key_layer.transpose(
                -1, -2))

            attention_scores = attention_scores / math.sqrt(
                self.hidden_size)

            if return_attention:
                return attention_scores

            # Normalize the attention scores to probabilities.
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

            # This is actually dropping out entire tokens to attend to, which might
            # seem a bit unusual, but is taken from the original Transformer paper.
            if is_drop:
                attention_probs = self.attn_dropout(attention_probs)
            # [cells, emb] = [cells, cells] * [cells, emb]
            context_layer = torch.matmul(attention_probs, value_layer)

            outputs.append(context_layer)
        # avg([heads, cells, emb])
        output = torch.mean(torch.stack(outputs), 0)

        hidden_states = self.dense(output)
        hidden_states = self.out_dropout(hidden_states)
        #         hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def save_model(self, file_name):
        torch.save(self.cpu().state_dict(), file_name)
        self.to(self.args.device)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

class ConstrastiveLoss(nn.Module):
    def __init__(self, cells_num, temperature):
        super(ConstrastiveLoss, self).__init__()
        self.cells_num = cells_num
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(cells_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, cells_num):
        N = 2 * cells_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(cells_num):
            mask[i, cells_num + i] = 0
            mask[cells_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.cells_num
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.cells_num)
        sim_j_i = torch.diag(sim, -self.cells_num)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

