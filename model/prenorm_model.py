import torch
import numpy as np
import torch.nn as nn
import math


# BERT Parameters
maxlen = 150 # vocab.json:maxlen 
# batch_size = 6
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768*4 # 4*d_model, FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 26 # todo

# # len(vocab.json: [0])
vocab_size = 26

device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    tmp = torch.ones(batch_size, 1).to(device)
    _seq_k = torch.cat((tmp, seq_k) ,dim=1)
    pad_attn_mask = _seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k + 1(=len_q + 1), one is masking
    return pad_attn_mask.expand(batch_size, len_q + 1, len_k + 1)  # batch_size x len_q+1 x len_k+1

def get_attn_subsequent_mask(seq):
    attn_shape = [seq.size(0), seq.size(1) + 1, seq.size(1) + 1]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask.to(device)

# def get_attn_pad_mask(seq_q, seq_k):
#     batch_size, seq_len = seq_q.size()
#     # eq(zero) is PAD token
#     tmp = torch.ones(batch_size, 1).to(device)
#     _seq_q = torch.cat((tmp, seq_q), dim=1)
#     pad_attn_mask = _seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len + 1]
#     return pad_attn_mask.expand(batch_size, seq_len + 1, seq_len + 1)  # [batch_size, seq_len+1, seq_len+1]

def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x).to(device)  # [seq_len] -> [batch_size, seq_len]
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.W_O = nn.Linear(n_heads * d_v, d_model)
        # self.Layernorm = nn.LayerNorm(d_model)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads, d_v]
        output = self.W_O(context)
        return output # output: [batch_size, seq_len, d_model]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.Layernorm1 = nn.LayerNorm(d_model)
        self.Layernorm2 = nn.LayerNorm(d_model)

    def forward(self, enc_inputs, enc_self_attn_mask):
        LN_enc_inputs_1 = self.Layernorm1(enc_inputs)
        enc_outputs_3 = enc_inputs + self.enc_self_attn(LN_enc_inputs_1, LN_enc_inputs_1, LN_enc_inputs_1, enc_self_attn_mask) # enc_inputs to same Q,K,V
        LN_enc_outputs_4 = self.Layernorm2(enc_outputs_3)
        enc_outputs_5 = self.pos_ffn(LN_enc_outputs_4) # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs_5 + enc_outputs_3

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.final_layernorm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 6)
        # self.linear = nn.Linear(d_model, d_model)
        # self.activ2 = gelu
        # # fc2 is shared with embedding layer
        # embed_weight = self.embedding.tok_embed.weight
        # self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        # self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, start_state):
        output = self.embedding(input_ids, segment_ids) # [bach_size, seq_len, d_model]
        output = torch.cat((output, start_state), dim=1)
        enc_self_attn_pad_mask = get_attn_pad_mask(input_ids, input_ids) # [batch_size, maxlen + 1, maxlen + 1]
        enc_self_attn_subsequent_mask = get_attn_subsequent_mask(input_ids)
        enc_self_attn_mask = torch.gt((enc_self_attn_pad_mask + enc_self_attn_subsequent_mask), 0)

        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        output = self.final_layernorm(output)
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output) # [batch_size, max_len+1, d_model]
        logits_clsf = self.classifier(h_pooled) #  batch_size x input_len (maxlen+1) x 6

        return logits_clsf

if __name__ == '__main__':
    seq = torch.tensor([[1, 3, 5, 0], [2, 2, 0, 0]]).to(device)
    attn_pad_mask = get_attn_pad_mask(seq ,seq)
    attn_subsequent_mask = get_attn_subsequent_mask(seq)
    print(attn_pad_mask, attn_subsequent_mask)
    dec_self_attn_mask = torch.gt((attn_pad_mask + attn_subsequent_mask), 0)
    print(dec_self_attn_mask)

    input_ids=  torch.ones(8, 4).long().to(device)
    segment_ids = torch.ones(8, 4).long().to(device)
    start_state = torch.ones(8, 6).to(device)
    tar_start_state = torch.zeros(8, 1, d_model).to(device)
    tar_start_state[:, 0, :6] = start_state
    model = BERT().to(device)

    input_ids[0,-1] = 0
    logits = model(input_ids, segment_ids, tar_start_state)
    print(logits.shape)  # # batch_size x input_len (maxlen+1) x 6