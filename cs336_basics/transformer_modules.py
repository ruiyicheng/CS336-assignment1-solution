import torch
from einops import rearrange, einsum, reduce, repeat
import numpy as np

############################# Linear

class Linear(torch.nn.Module):
    def __init__(self,in_features, out_features, device=None, dtype=None):
        super().__init__()
        W = torch.empty(out_features,in_features,dtype = dtype,device = device)
        std = np.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(W, mean=0.0, std=1, a=-3, b=3.0, generator=None)
        W = W * std
        self.W = torch.nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #print(x.shape)
        #print(self.W.shape)
        return einsum(self.W,x,'out_feature in_feature, ... in_feature -> ... out_feature')
    
############################# Embedding
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embedding_matrix = torch.empty(num_embeddings,embedding_dim,device = device,dtype = dtype)
        torch.nn.init.trunc_normal_(embedding_matrix, mean=0.0, std=1, a=-3, b=3.0, generator=None)
        self.embedding_matrix = torch.nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]

############################# RMSNorm
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        gain = torch.ones(d_model,device = device, dtype = dtype)
        self.gain = torch.nn.Parameter(gain)
        self.d_model = d_model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        a_i_g_i = einsum(x,self.gain,'batch_size sequence_length d_model, d_model -> batch_size sequence_length d_model')
        dominator = torch.sqrt(einsum(x,x,'batch_size sequence_length d_model, batch_size sequence_length d_model -> batch_size sequence_length')/self.d_model + self.eps)
        result = einsum(a_i_g_i,1/dominator,"batch_size sequence_length d_model, batch_size sequence_length -> batch_size sequence_length d_model")

        return result.to(in_dtype)

############################# positionwise_feedforward
class positionwise_feedforward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        std_initiate = np.sqrt(2/(d_model+d_ff))
        w1 = torch.empty(d_ff,d_model, device = device, dtype = dtype)
        w3 = torch.empty(d_ff,d_model, device = device, dtype = dtype)
        w2 = torch.empty(d_model,d_ff, device = device, dtype = dtype)
        torch.nn.init.trunc_normal_(w1, mean=0.0, std=1, a=-3, b=3.0, generator=None)
        torch.nn.init.trunc_normal_(w2, mean=0.0, std=1, a=-3, b=3.0, generator=None)
        torch.nn.init.trunc_normal_(w3, mean=0.0, std=1, a=-3, b=3.0, generator=None)
        w1 = w1 * std_initiate
        w2 = w2 * std_initiate
        w3 = w3 * std_initiate
        self.W1 = torch.nn.Parameter(w1)
        self.W2 = torch.nn.Parameter(w2)
        self.W3 = torch.nn.Parameter(w3)

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        W1x = einsum(self.W1,x,"d_ff d_model, ... d_model -> ... d_ff")
        Silu_W1x = einsum(W1x,torch.sigmoid(W1x),"... d_ff, ... d_ff -> ... d_ff")
        W3x = einsum(self.W3,x,"d_ff d_model, ... d_model -> ... d_ff")
        Silu_W1x_W3x = einsum(Silu_W1x,W3x,"... d_ff, ... d_ff -> ... d_ff")
        SwiGLU = einsum(self.W2,Silu_W1x_W3x,"d_model d_ff, ... d_ff -> ... d_model")
        return SwiGLU


############################# RotaryPositionalEmbedding
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        i_range = torch.arange(max_seq_len, device = device)
        k_range = torch.arange(start = 0, end = d_k//2, device = device)
        theta_ik = einsum(i_range,1/torch.pow(theta,2*k_range/d_k), "i, k -> i k")
        cos_theta = torch.cos(theta_ik)
        sin_theta = torch.sin(theta_ik)

        #self.register_buffer("cos_theta",cos_theta,persistent=False)
        #self.register_buffer("sin_theta",sin_theta,persistent=False)

        # print(sin_theta.shape)
        R_ik_upper = torch.stack((cos_theta,-sin_theta),dim = 0)
        R_ik_lower = torch.stack((sin_theta,cos_theta),dim = 0)
        R_ik = torch.stack((R_ik_upper,R_ik_lower),dim = 0)
        R_ik = rearrange(R_ik,'row col i k -> i k row col')
        #print(R_ik.shape) # it should be seq_len, dk(embedding_dim)//2,2,2
        self.register_buffer("R_ik",R_ik,persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: ((batch) seq_length dk)
        mat_used = self.R_ik[token_positions,:,:,:] 
        #mat_used ((batch) seq_length dk//2 2 2)
        x_in = rearrange(x,"... seq_length (dk_over_two two) -> ... seq_length dk_over_two two",two = 2) 
        #x_in ((batch) seq_length dk//2 2)
        # print(mat_used.shape)
        rope_result = einsum(mat_used,x_in,"... seq_length dk_2 two_row two_col, ... heads seq_length dk_2 two_col -> ... heads seq_length dk_2 two_row")
        rope_result_flatted = rearrange(rope_result, "... seq_length dk_2 two_row -> ... seq_length (dk_2 two_row)")
        return rope_result_flatted
# rope = RotaryPositionalEmbedding(10000,16,64)

############################# softmax
def softmax(x: torch.Tensor,ith_dim: int):
    
    max_input = torch.max(x, ith_dim, keepdim=True).values
    # print(type(max_input),type(x))
    input_reduced = x - max_input
    input_reduced_exp = torch.exp(input_reduced)
    result = input_reduced_exp/torch.sum(input_reduced_exp, ith_dim, keepdim=True)
    return result

############################# scaled_dot_product_attention
def scaled_dot_product_attention(Q: torch.Tensor,K: torch.Tensor,V: torch.Tensor,mask: torch.Tensor = None,device = None):
    # Shape of Q and K: (batch_size, ..., seq_len, d_k)
    # Shape of V: (batch_size, ..., seq_len, d_v)
    # softmax(Q^T K/sqrt(d_k) ) V
    d_k = K.shape[-1]
    QT_K_sqrt_d = einsum(Q,K,"batch_size ... query d_k, batch_size ... key d_k -> batch_size ... query key")/np.sqrt(d_k)
    if not mask is None:
        mask_used = torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf'))).to(device = device)
        #print(device)
        #print(mask_used.device)
        QT_K_sqrt_d = QT_K_sqrt_d + mask_used
    #print("QT_K_sqrt_d")
    #print(QT_K_sqrt_d)
    softmax_QT_K_sqrt_d = softmax(QT_K_sqrt_d,-1)
    attention = einsum(softmax_QT_K_sqrt_d,V,"batch_size ... query key, batch_size ... key d_v -> batch_size ... query d_v")
    return attention

############################# multihead_self_attention
class multihead_self_attention(torch.nn.Module):
    def __init__(self,d_model: int, num_heads: int,max_seq_len: int | None = None,theta: float|None = None,device = None, dtype = None,use_RoPE = False):
        super().__init__()
        #print('device for attention:',device)
        self.device = device
        self.use_RoPE = use_RoPE
        d_v = d_model//num_heads
        d_k = d_model//num_heads
        self.dv = d_v
        self.dk = d_k
        W_Q = torch.empty(d_k * num_heads,d_model,device = device, dtype = dtype)
        W_K = torch.empty(d_k * num_heads,d_model,device = device, dtype = dtype)
        W_V = torch.empty(d_v * num_heads,d_model,device = device, dtype = dtype)
        W_O = torch.empty(d_model,d_v * num_heads,device = device, dtype = dtype)

        std = np.sqrt(2/(d_v * num_heads + d_model))

        torch.nn.init.trunc_normal_(W_Q, mean=0.0, std=1, a=-3, b=3.0, generator=None)
        torch.nn.init.trunc_normal_(W_K, mean=0.0, std=1, a=-3, b=3.0, generator=None)
        torch.nn.init.trunc_normal_(W_V, mean=0.0, std=1, a=-3, b=3.0, generator=None)
        torch.nn.init.trunc_normal_(W_O, mean=0.0, std=1, a=-3, b=3.0, generator=None)

        W_Q = W_Q * std
        W_K = W_K * std
        W_V = W_V * std
        W_O = W_O * std

        self.W_Q = torch.nn.Parameter(W_Q)
        self.W_K = torch.nn.Parameter(W_K)
        self.W_V = torch.nn.Parameter(W_V)
        self.W_O = torch.nn.Parameter(W_O)
        if self.use_RoPE:
            self.RoPE = RotaryPositionalEmbedding(theta,d_k,max_seq_len,device = device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:

        # x (... sequence_length d_model)
        # token_position (... sequence_length)
        # QK -> RoPE
        # print('shape of x is', x.shape)
        # print(x)
        # token_positions = None
        if token_positions is None and self.use_RoPE:
            token_positions = torch.arange(x.shape[-2],device=x.device)
            for _ in range(x.ndim - 2):
                token_positions = token_positions.unsqueeze(0)
            token_positions = token_positions.expand(*x.shape[:-1])

        Q_origin = einsum(self.W_Q,x,"d_k_num_heads d_model, ... sequence_length d_model -> ... sequence_length d_k_num_heads")
        K_origin = einsum(self.W_K,x,"d_k_num_heads d_model, ... sequence_length d_model -> ... sequence_length d_k_num_heads")
        V_origin = einsum(self.W_V,x,"d_v_num_heads d_model, ... sequence_length d_model -> ... sequence_length d_v_num_heads")
        Q_ready_for_transformer = rearrange(Q_origin,"... sequence_length (num_heads dk) -> ... num_heads sequence_length dk",dk = self.dk)
        K_ready_for_transformer = rearrange(K_origin,"... sequence_length (num_heads dk) -> ... num_heads sequence_length dk",dk = self.dk)
        V_ready_for_transformer = rearrange(V_origin,"... sequence_length (num_heads dv) -> ... num_heads sequence_length dv",dv = self.dv)
        if self.use_RoPE:

            Q_ready_for_transformer = self.RoPE(Q_ready_for_transformer,token_positions = token_positions)
            K_ready_for_transformer = self.RoPE(K_ready_for_transformer,token_positions = token_positions)

        # reshape Q K V to enable attention calculation

        Q_length = Q_ready_for_transformer.shape[-2]
        K_length = K_ready_for_transformer.shape[-2]
        mask = ~torch.triu(torch.ones(Q_length, K_length,dtype = bool), diagonal=1)
        # print(Q_ready_for_transformer)
        # print(K_ready_for_transformer)
        # print(V_ready_for_transformer)
        # print(mask)

        attention = scaled_dot_product_attention(Q_ready_for_transformer,K_ready_for_transformer,V_ready_for_transformer, mask = mask,device = self.device)
        # shape = batch_size ... num_heads len_query d_v
        # print(attention)
        attention_concat = rearrange(attention,"... num_heads len_query dv-> ... (num_heads dv) len_query")
        # print("shape of attention is",attention.shape)
        #print("shape of attention_concat is",attention_concat.shape )
        #print("shape of W_O is",self.W_O.shape)

        multihead_attention = einsum(self.W_O,attention_concat,"d_model num_headsdv, ... num_headsdv len_query  -> ... len_query d_model")
        #print(multihead_attention)
        return multihead_attention



############################# transformer_block

class transformer_block(torch.nn.Module):
    def __init__(self,d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device = None, dtype = None):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attention = multihead_self_attention(d_model,num_heads,max_seq_len = max_seq_len, theta = theta,device = device, dtype = dtype, use_RoPE = True)
        self.norm2 = RMSNorm(d_model)
        self.feed_forward = positionwise_feedforward(d_model,d_ff)
        self.device = device
    def forward(self,x : torch.Tensor) -> torch.Tensor:
        y = x + self.attention(self.norm1(x))
        z = y + self.feed_forward(self.norm2(y))
        return z

############################# transformer_lm
class transformer_lm(torch.nn.Module):
    def __init__(self,d_model: int, num_heads: int, d_ff: int, vocab_size: int, num_layers: int,max_seq_len : int,theta : float ,device = None, dtype = None):
        super().__init__()
        self.device = device
        self.embedding = Embedding(vocab_size,d_model,dtype = dtype, device = device)
        self.transformers = torch.nn.ModuleList([transformer_block(d_model, num_heads, d_ff, max_seq_len,theta,device = device, dtype = dtype) for _ in range(num_layers)])
        self.norm_final = RMSNorm(d_model,device = device, dtype = dtype)
        self.final_linear = Linear(d_model,vocab_size,device = device, dtype = dtype)
        #print(len(self.transformers))
    def forward(self,x: torch.Tensor):
        x = self.embedding(x)
        # print(x.shape)
        for transformer_this in self.transformers:
            x = transformer_this(x)
            # print('l')
            # print(x.shape)
        x = self.norm_final(x)
        # print(x.shape)
        x = self.final_linear(x)
        # print(x.shape)
        #x = softmax(x,-1)
        #print(x.shape)
        #print(x)
        return x





        

def decoding_one(logits:torch.Tensor,temperature=1,top_p= None):
    eps = 1e-3
    temperature = np.maximum([temperature,eps])
    logits = logits/temperature
    if top_p is not None:
        logits_sort = torch.sort(logits,axis = -1,descending=False)
        threshold = logits_sort[top_p]
        logits[logits<threshold] = logits[logits<threshold] - 100000
    prob = softmax(logits)
    chosen_index = torch.multinomial(prob, num_samples=1)
    return chosen_index
    


def decoding(model,input_tokens,vocab_dict,max_tokens,temperature=1,top_p= None):
    input_tokens_this = input_tokens.clone()
    for _ in range(max_tokens):
        logits = model(input_tokens_this)
        logits = logits[-1,:]  # Get the logits for the last token
        chosen_index = decoding_one(logits,temperature = temperature,top_p=top_p)
        input_tokens_this = torch.cat((input_tokens_this,chosen_index),dim=-1)
        # remove the first token if input token
        input_tokens_this = input_tokens_this[1:]
        if vocab_dict[int(chosen_index.item())] == "<|end_of_text|>":
            break
    # translate it into text using vocab_dict
    output_text = []
    for token in input_tokens_this[0]:
        output_text.append(vocab_dict[int(token.item())])
    output_text = "".join(output_text)
    return output_text, input_tokens_this


    