# Transformer
## Transformer系列的火热
用Transformer模型结构进行大规模语言模型（language model）预训练（Pre-train），再在多个NLP下游（downstream）任务中进行微调（Finetune），不同的结构改进和训练方式改进如下图

![transformer系列的改进](./img/1-x-formers.png)

## Transformer的模型结构
![transformer模型结构](./img/2-transformer.png)
### Transformer的宏观结构
可以视作一种seq2seq模型，由多层encoder和多层decoder组成。
如图 ![deocder和encoder的图](./img/2-2-encoder-detail.png)
#### Enocder
单层encoder由两个子层组成，分别是多头自注意力机制（multi-head self-attention mechanism）和前馈神经网络（feed-forward neural network）。
#### Decoder
单层decoder由三个子层组成，分别是多头自注意力机制（multi-head self-attention mechanism）、多头交叉注意力机制（multi-head cross-attention mechanism）和前馈神经网络（feed-forward neural network）。

总结一下，我们基本了解了Transformer由编码部分和解码部分组成，而编码部分和解码部分又由多个网络结构相同的编码层和解码层组成。每个编码层由self-attention和FFNN组成，每个解码层由self-attention、FFN和encoder-decoder attention组成。

### Transformer的细节、
了解了Transformer的宏观结构之后。下面，让我们来看看Transformer如何将输入文本序列转换为向量表示，又如何逐层处理这些向量表示得到最终的输出。
#### 输入向量
文本序列对应词汇表中的索引序列，通过embedding层将索引序列转换为向量表示，然后输入到encoder中逐层处理，得到最终的输出。

具体来说，设文本序列为 $$ \mathbf{x} = [x_1, x_2, \ldots, x_T]，$$ 其中 $x_i$为词汇表中的索引。通过 embedding 层，可以得到对应的向量表示 $$ \mathbf{e} = [e_1, e_2, \ldots, e_T] $$ ，其中 $ e_i = \mathbf{W}_{\text{emb}} x_i $ ，$ \mathbf{W}_{\text{emb}}$  为 embedding 矩阵，矩阵大小为$vocab\_size \times d_{model}$，其中$vocab\_size$是词汇表大小，$d_{model}$是词向量维度。

与其他seq2seq模型类似，我们使用学习到的embedding将输入token和输出token转换为$d_{\text{model}}$维的向量。我们还使用普通的线性变换和softmax函数将解码器输出转换为预测的下一个token的概率 在我们的模型中，两个嵌入层之间和pre-softmax线性变换共享相同的权重矩阵，类似于[cite](https://arxiv.org/abs/1608.05859)。在embedding层中，我们将这些权重乘以$\sqrt{d_{\text{model}}}$。
```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```
但是由于这些词向量是并行处理的，无法有效利用词语词之间的位置信息，因此，添加位置信息是十分有必要的。

在 Transformer 中，位置编码（positional encoding）被添加到词向量中，以提供每个词在句子中的位置信息。位置编码的公式如下：
$$
\text{PE}_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{\text{model}}}) \quad \text{and} \quad \text{PE}_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{\text{model}}})
$$
其中，$pos$ 表示位置，$i$ 表示维度，$d_{\text{model}}$ 表示词向量的维度。

通过这种方式，每个位置编码向量都包含了一个独特的模式，使得模型能够学习到位置信息。
我们选择这个函数是因为我们假设它会让模型很容易学习对相对位置的关注，因为对任意确定的偏移$k$, $PE_{pos+k}$ 可以表示为 $PE_{pos}$的线性函数。

  此外，我们会将编码器和解码器堆栈中的embedding和位置编码的和再加一个dropout。对于基本模型，我们使用的dropout比例是$P_{drop}=0.1$。
```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
```
#### Encoder Layer
编码器由N个相同的层组成，每一层包含两个子层，一个是多头自注意力机制，另一个是前馈神经网络。这两个子层都使用了残差连接，并在子层之后使用了层归一化。

```python
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Apply one encoder layer and return attention weights."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```
我们称呼子层为：$\mathrm{Sublayer}(x)$，每个子层的最终输出是$\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$。 dropout被加在Sublayer上。

为了便于进行残差连接，模型中的所有子层以及embedding层产生的输出的维度都为 $d_{\text{model}}=512$。

下面的SublayerConnection类用来处理单个Sublayer的输出，该输出将继续被输入下一个Sublayer：
```python
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
```

层标准化：
```python
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```
Attention中
$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$
$Q, K, V$均是输入向量$A$的线性变换，$$ Q = W^qA , K = W^kA, V = W^vA$$  
```python
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```
除了缩放因子$\frac{1}{\sqrt{d_k}}$ ，点积Attention跟我们的平时的点乘算法一样。加法attention使用具有单个隐层的前馈网络计算相似度。虽然理论上点积attention和加法attention复杂度相似，但在实践中，点积attention可以使用高度优化的矩阵乘法来实现，因此点积attention计算更快、更节省空间。
当$d_k$ 的值比较小的时候，这两个机制的性能相近。当$d_k$比较大时，加法attention比不带缩放的点积attention性能好 (cite)。我们怀疑，对于很大的$d_k$值, 点积大幅度增长，将softmax函数推向具有极小梯度的区域。(为了说明为什么点积变大，假设q和k是独立的随机变量，均值为0，方差为1。那么它们的点积$q \cdot k = \sum_{i=1}^{d_k} q_ik_i$, 均值为0方差为$d_k$)。为了抵消这种影响，我们将点积缩小 $\frac{1}{\sqrt{d_k}}$倍。

在此引用苏剑林文章《浅谈Transformer的初始化、参数化与标准化》中谈到的，为什么Attention中除以$\sqrt{d}$这么重要。

Multi-head attention允许模型同时关注来自不同位置的不同表示子空间的信息，如果只有一个attention head，向量的表示能力会下降。

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ..., \mathrm{head_h})W^O \
\text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

其中映射由权重矩阵完成：$W^Q_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^K_i \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W^V_i \in \mathbb{R}^{d_{\text{model}} \times d_v}$ and $W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$。

在这项工作中，我们采用$h=8$个平行attention层或者叫head。对于这些head中的每一个，我们使用$d_k=d_v=d_{\text{model}}/h=64$。由于每个head的维度减小，总计算成本与具有全部维度的单个head attention相似。
```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
```

multi-head attention在Transformer中有三种不同的使用方式：

在encoder-decoder attention层中，queries来自前面的decoder层，而keys和values来自encoder的输出。这使得decoder中的每个位置都能关注到输入序列中的所有位置。这是模仿序列到序列模型中典型的编码器—解码器的attention机制，例如 (cite).
encoder包含self-attention层。在self-attention层中，所有key，value和query来自同一个地方，即encoder中前一层的输出。在这种情况下，encoder中的每个位置都可以关注到encoder上一层的所有位置。
类似地，decoder中的self-attention层允许decoder中的每个位置都关注decoder层中当前位置之前的所有位置（包括当前位置）。 为了保持解码器的自回归特性，需要防止解码器中的信息向左流动。我们在缩放点积attention的内部，通过屏蔽softmax输入中所有的非法连接值（设置为$-\infty$）实现了这一点。
#### Decoder Layer
解码器由N个相同的层组成，每一层包含三个子层，分别是多头自注意力机制、多头交叉注意力机制和前馈神经网络。这三个子层都使用了残差连接，并在子层之后使用了层归一化。

单层decoder与单层encoder相比，decoder还有第三个子层，该层对encoder的输出执行attention：即encoder-decoder-attention层，q向量来自decoder上一层的输出，k和v向量是encoder最后层的输出向量。与encoder类似，我们在每个子层再采用残差连接，然后进行层标准化。
```python
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
```
对于单层decoder中的self-attention子层，我们需要使用mask机制，以防止在当前位置关注到后面的位置。
```python
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```

#### Encoder
编码器由N = 6个完全相同的层组成。
```python
def clones(module, N):
    "产生N个完全相同的网络层"
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    "完整的Encoder包含N层"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "每一层的输入是x和mask"
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
```
#### Decoder        
解码器也是由N = 6 个完全相同的decoder层组成。
```python
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
```
大部分序列到序列（seq2seq）模型都使用编码器-解码器结构 (引用)。编码器把一个输入序列$(x_{1},...x_{n})$映射到一个连续的表示$z=(z_{1},...z_{n})$中。解码器对z中的每个元素，生成输出序列$(y_{1},...y_{m})$。解码器一个时间步生成一个输出。在每一步中，模型都是自回归的(引用)，在生成下一个结果时，会将先前生成的结果加入输入序列来一起预测。EncoderDecoder类来搭建一个seq2seq架构：
```python
class EncoderDecoder(nn.Module):
    """
    基础的Encoder-Decoder结构。
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
class Generator(nn.Module):
    "定义生成器，由linear和softmax组成"
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```

### 完整模型
```python
def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model
```

```python
# Small example model.
tmp_model = make_model(10, 10, 2)
```