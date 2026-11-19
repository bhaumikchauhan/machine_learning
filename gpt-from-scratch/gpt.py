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
        self.wei = wei # (B, T, T)
        x = self.value(x) # (B, T, head_size)
        x = wei @ x # (B, T, head_size)
        # (B, T, T) @ (B, T, head_size) 
        # B remains same so
        # (T, T) @ (T, head_size), results in (T, head_size)
        return x
         
    


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=2)


class FeedForward(nn.Module):
    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.blocks = nn.Sequential(Block(n_embd, 4), Block(n_embd, 4), Block(n_embd, 4))

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
        #x = self.sa_heads(x)
        # each head returns (B, T, head_size)
        # then MultiHeadAttention concatenates over last dimension
        # so x.shape is (B, T, 4*head_size)
        # head_size is n_embd//4, x becomes (B, T, n_embd)
        # That only works because 32 divides evenly by 4. 
        # With n_embd = 30 and 4 heads, 30 // 4 is 7, and 4 × 7 = 28. 
        # x would come out as (B, T, 28), and lm_head's Linear(30, 65) would fail
        #x = self.ffwd(x)
        x = self.blocks(x)
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



for i, head in enumerate(model.blocks[0].sa.heads):
    wei = head.wei
    assert wei.shape == (BATCH_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    assert torch.allclose(wei.sum(dim=-1), torch.ones(BATCH_SIZE, BLOCK_SIZE, device=DEVICE))
    assert (torch.triu(wei, diagonal=1) == 0).all()
    print(f"wei of the first sequence in the batch: {i}")
    print(wei[0].round(decimals=2))

#calculate all the weights:
# layers: 
weights_in_token_embedding = vocab_size * n_embd
weights_in_poistion_embedding = BLOCK_SIZE * n_embd
weights_in_Key_in_head = n_embd * 8
weights_in_Query_in_head = n_embd * 8
weights_in_Value_in_head = n_embd * 8
number_of_heads = 4
all_weights_in_all_heads = number_of_heads * (weights_in_Key_in_head + weights_in_Query_in_head + weights_in_Value_in_head)
num_blocks = 3
weights_and_bias__in_ffwd = n_embd * n_embd + n_embd
weights_in_one_block = all_weights_in_all_heads + weights_and_bias__in_ffwd
weights_in_all_blocks = num_blocks * weights_in_one_block
weights_in_lm_head = n_embd * vocab_size + vocab_size

total_number_of_weights = weights_in_token_embedding + weights_in_poistion_embedding + weights_in_all_blocks + weights_in_lm_head

assert sum(p.numel() for p in model.parameters()) == total_number_of_weights


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