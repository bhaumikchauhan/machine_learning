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


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx) # (B,T,n_embd)
        logits = self.lm_head(tok_emb)

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
            logits, _ = self(idx)
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


#assert logits.shape == (4,8,65)

assert vocab_size == 65
assert len(data) == 1115394
assert decode(encode("hi there")) == "hi there"
assert len(train_data) == 1003854
assert len(validation_data) == 111540


# assert x.shape == (4,8)
# assert y.shape == (4,8)
# assert torch.equal(x[:, 1:], y[:, :-1])