import torch


torch.manual_seed(1337)
BATCH_SIZE=4
BLOCK_SIZE=8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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


x, y = get_batch('train')

assert vocab_size == 65
assert len(data) == 1115394
assert decode(encode("hi there")) == "hi there"
assert len(train_data) == 1003854
assert len(validation_data) == 111540


assert x.shape == (4,8)
assert y.shape == (4,8)
assert torch.equal(x[:, 1:], y[:, :-1])
print(decode(x[0].tolist()))
print(decode(y[0].tolist()))