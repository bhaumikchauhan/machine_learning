import torch


torch.manual_seed(1337)

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
training_data_percent = 0.9

train_data = data[:int(training_data_percent*len(data))]
validation_data = data[int(training_data_percent*len(data)):]

assert vocab_size == 65
assert len(data) == 1115394
assert decode(encode("hi there")) == "hi there"
assert len(train_data) == 1003854
assert len(validation_data) == 111540