import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
block_size = 8
max_iters = 300
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
head_size = 16

with open ('input.txt', 'r') as f:
    text = f.read()
print(len(text))

cahrs = sorted(list(set(text)))
vocab_size = len(cahrs)
print(vocab_size)

#create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(cahrs)}
itos = {i:ch for i, ch in enumerate(cahrs)}
encoder = lambda s: [stoi[c] for c in s]# encoder: takes a string and outputs a list of integers
decoder = lambda l: ''.join(itos[i] for i in l)#decoder: takes a list of integers and outputs a string

print(encoder('Pouyan'))
print(decoder([28, 53, 59, 63, 39, 52]))

#incoding the entire text dataset and store it in a tensor
data = torch.tensor(encoder(text), dtype=torch.long)

#splitting the data into train and validation sets
n = int (0.9* len(data))#first 90% of the data set will be used in training
train_data = data[:n]
val_data = data[n:]#validating like this will help us understand how much our model is over fitting



x = train_data[:block_size]
y = train_data[1:block_size+1]
for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    print(f"when the context is{context}, the target is [{target}]")

#intoducing batch, with that we can feed multiple text batches there for in GPU the transformer can process multiple batches simultaneously
torch.manual_seed(1337)


def get_batch(split):
    # generate a small batch of data of inputs x and  targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size ]for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x , y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits , loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

Xb , Yb = get_batch('train')
print('inputs:')
print(Xb.shape)
print(Xb)

print('targets:')
print(Yb.shape)
print(Yb)

for b in range(batch_size):
    for t in range(block_size):
        context = Xb[b , :t +1]
        target = Yb[b , t]
        print(f"when the context is{context}, the target is [{target}]")

torch.manual_seed(1337)

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size,bias=False)
        self.query = nn.Linear(n_embd, head_size,bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5# (B,T,16) @ (B,16,T) ---->(B,T,T)
        wei = wei.masked_fill(self.tril [:T,:T] == 0 , float('-inf'))# Decoder block makes sure there's no communication with future
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v# (B,T,T) @ (B,T,C) ----> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads, head_size):
        super().__init__()
        self.head = nn.ModuleList([Head(head_size)for _ in range (num_heads)])

    def forward(self,x):
        return torch.cat([h(x) for h in self.head], dim = -1)

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(n_embd,n_embd),
            nn.ReLU()
        )
    def forward(self,x):
        return self.net(x)

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        #each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4,n_embd//4)# 4 of 8 dimensional self attention
        self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target = None):
        B, T = idx.shape
        #idx and target are both (B,T) tensors of integers]
        token_emb  = self.token_embedding_table(idx)#(B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.sa_heads(x)#appliying Multiple head of self attention. (B,T,C)
        x = self.ffwd(x) #(B,T,C)
        logits = self.lm_head(x)#(B,T,vocab_size)
        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # crop idx to the last blok_size tokens
            idx_cond = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cond)
            # focus on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, 1)
            # append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx




model = BigramLanguageModel()
m = model.to(device)
out = m(Xb,Yb)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
batch_size = 32
steps = 10000
for step in range(steps):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step: {step}, Train loss: {losses['train']:.4f}, Validation loss: {losses['val']:.4f}")
    #sample a batch of data
    Xb , Yb = get_batch('train')
    #evaluate the loss
    logits, loss = m(Xb,Yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# print(loss.item())
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decoder(m.generate(context,max_new_tokens = 500)[0].tolist()))

# #Self-Attention
# B,T,C = 4,8,32# Batch, Time, Channel
# x = torch.randn((B,T,C))
#
# #Head of self attention
# head_size = 16
# key = nn.Linear(C,head_size, bias=False)
# query = nn.Linear(C,head_size, bias=False)
# value = nn.Linear(C, head_size, bias=False)
# k = key(x) #(B, T , 16)
# q = query(x)#(B, T , 16)
# v = value(x)#(B, T , 16)
# wei = q @ k.transpose(-2,-1)#(B,T.16) @ (B,16 ,T) ----> (B,T,T)
#
# tril = torch.tril(torch.ones(T,T))
# # wei = torch.zeros(T,T)
# wei = wei.masked_fill(tril == 0,float('-inf'))
# wei = F.softmax(wei, dim = -1)
# out = wei @ v
# print(f'Weights = {wei}')
# torch.manual_seed(42)
# a = torch.tril(torch.randn((3,3),device=device))
# a = torch.abs(a)/ torch.sum(torch.abs(a),1,keepdim=True)
# b = torch.randint(0,10,(3,2)).float()
# c = a @ b
# print(f'A = {a}         B = {b}        C = {c}')