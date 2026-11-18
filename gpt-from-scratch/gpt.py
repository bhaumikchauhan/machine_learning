from typing import Any

from torch.nn import functional as F
import torch
import torch.nn as nn


torch.manual_seed(1337)
BATCH_SIZE=32
BLOCK_SIZE=8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR=1e-3
ITERATIONS=5000
EVAL_INTERVAL=500
EVAL_ITERS=200
n_embd = 32

with open("input.txt", "r", encoding="utf-8") as f:
    input_text = f.read()
    
chars = sorted(list(set(input_text)))
vocab_size = len(chars)
print(f"Number of characters in input.txt: {len(input_text)}")
print(f"Number of unique characters: {len(chars)}")
print(f"Unique characters: {chars}")


stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = torch.tensor(encode(input_text), dtype=torch.long)
train_fraction = 0.9

train_data = data[:int(train_fraction*len(data))]
validation_data = data[int(train_fraction*len(data)):]


def get_batch(split):
    data = train_data if split == 'train' else validation_data
    x = []
    y = []
    start_indices = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    for i in range(BATCH_SIZE):
        x.append(data[start_indices[i]:start_indices[i]+BLOCK_SIZE])
        y.append(data[start_indices[i]+1:start_indices[i]+BLOCK_SIZE+1])

    x = torch.stack(x).to(DEVICE)
    y = torch.stack(y).to(DEVICE)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B,T,n_embed = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.wei = wei        
        return wei @ self.value(x)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.sa_head = Head(n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        # x is (B, T): each value is an id, used as a row number into the token table (65 x 32),
        # so tok_emb is (B, T, 32).
        # torch.arange(T) gives the ids 0..T-1, the row numbers into the position table (8 x 32),
        # so pos_emb is (T, 32).
        # tok_emb + pos_emb: row t of the position table is added to the token vector at position t,
        # in every one of the B blocks (broadcasting), giving (B, T, 32).
        pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE)) # (T,n_embd)
        x = tok_emb + pos_emb # (B,T,n_embd) + (T,n_embd) = B, T, n_embd
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flattened = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits_flattened, targets)

        return logits, loss
        

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            last_time_step = logits[:,-1,:] #B,C
            probs = F.softmax(last_time_step, dim=1) #B,C
            next_id = torch.multinomial(probs, num_samples=1) #B,1
            idx = torch.cat((idx, next_id), dim=1)
        return idx


model = BigramLanguageModel(vocab_size)
model = model.to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)




for i in range(ITERATIONS+1):
    if i % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        print(f"step {i}: training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

idx = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
out = model.generate(idx, 500)
out_str = decode(out[0].tolist())

print("output:")
print(out_str)


# Check: the attention weights of one batch. Each row of wei sums to 1, and everything above the diagonal is 0.
with torch.no_grad():
    xb, yb = get_batch('train')
    model(xb, yb)  # runs Head.forward, which stores wei (B, T, T)
wei = model.sa_head.wei
assert wei.shape == (BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE)
assert torch.allclose(wei.sum(dim=-1), torch.ones(BATCH_SIZE, BLOCK_SIZE, device=DEVICE))
assert (torch.triu(wei, diagonal=1) == 0).all()
print("wei of the first sequence in the batch:")
print(wei[0].round(decimals=2))

#run the model on a batch, change only the last character of each row, run it again. 
# The logits at every position except the last must be identical (`torch.equal`). 
# If they change, the mask isn't working.

with torch.no_grad():
    xb, yb = get_batch('train')
    logits_1, loss = model(xb)

    # A copy, so xb keeps the original. Change the last character of every block:
    # id // 2 is always a different id, except for 0, which becomes 1.
    xb2 = xb.clone()
    last = xb[:, -1]  # (B,): the last character of each block
    xb2[:, -1] = torch.where(last != 0, last // 2, 1)
    assert (xb2[:, -1] != xb[:, -1]).all()  # every block's last character changed
    assert torch.equal(xb2[:, :-1], xb[:, :-1])  # and nothing else

    logits_2, loss = model(xb2)

    # Positions 0 to 6 can't see position 7, so their logits must not move at all.
    assert torch.equal(logits_1[:, :-1], logits_2[:, :-1])
    # Position 7 sees its own new character, so in every block its 65 logits must differ somewhere.
    assert (logits_1[:, -1] != logits_2[:, -1]).any(dim=-1).all()





#assert logits.shape == (4,8,65)

assert vocab_size == 65
assert len(data) == 1115394
assert decode(encode("hi there")) == "hi there"
assert len(train_data) == 1003854
assert len(validation_data) == 111540


# assert x.shape == (4,8)
# assert y.shape == (4,8)
# assert torch.equal(x[:, 1:], y[:, :-1])