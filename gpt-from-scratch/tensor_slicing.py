"""Tensor dimensions, axes and slicing, with small examples and the patterns used in gpt.py.

Run it and read the output next to the code: python tensor_slicing.py
Each section prints the code it ran, the result and the result's shape.
"""
import torch


def show(code, result):
    """Prints a line of code, then its result and shape."""
    print(f">>> {code}")
    if isinstance(result, torch.Tensor):
        print(result)
        print(f"    shape {tuple(result.shape)}")
    else:
        print(result)
    print()


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


# ----------------------------------------------------------------------
section("1. Dimensions and shape")
# A dimension (also called an axis) is one direction you can index along.
# The shape lists the size of each dimension, from the outermost to the innermost.
# Dimension 0 is the outermost. Dimension -1 is always the last one, -2 the one before it.

scalar = torch.tensor(7)
vector = torch.tensor([10, 11, 12])
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
show("scalar.ndim, scalar.shape", (scalar.ndim, tuple(scalar.shape)))
show("vector.ndim, vector.shape", (vector.ndim, tuple(vector.shape)))
show("matrix.ndim, matrix.shape", (matrix.ndim, tuple(matrix.shape)))

# The tensor used for most of this file: B=2 blocks, T=3 positions, C=4 channels, like tok_emb.
# The values are 0..23, so each value tells you where it sits: value = b*12 + t*4 + c.
x = torch.arange(24).view(2, 3, 4)
show("x = torch.arange(24).view(2, 3, 4)   # (B, T, C)", x)
print("Read it as: 2 blocks; each block has 3 rows (positions); each row has 4 numbers (channels).")
print("In Java terms: int[2][3][4] x. x[b][t][c] is one number.\n")

# ----------------------------------------------------------------------
section("2. An integer index removes that dimension; a slice keeps it")

show("x[0]          # block 0: the first dimension is gone", x[0])
show("x[0, 1]       # block 0, position 1: two dimensions gone", x[0, 1])
show("x[0, 1, 2]    # one number: all three dimensions gone", x[0, 1, 2])
show("x[0:1]        # a slice of length 1: the dimension stays, with size 1", x[0:1])
print("x[0] is (3, 4) but x[0:1] is (1, 3, 4): the same numbers, a different shape.\n")

# ----------------------------------------------------------------------
section("3. Slicing one dimension: start:stop:step")
# Exactly like Python lists: start is included, stop is excluded, negative counts from the end.

v = torch.arange(8)
show("v = torch.arange(8)", v)
show("v[2:5]    # indices 2, 3, 4 (stop excluded)", v[2:5])
show("v[:3]     # the first 3", v[:3])
show("v[-3:]    # the last 3", v[-3:])
show("v[:-1]    # everything except the last", v[:-1])
show("v[1:]     # everything except the first", v[1:])
show("v[::2]    # every second one", v[::2])
show("v[-1]     # the last one (an integer index: no dimension left)", v[-1])

# ----------------------------------------------------------------------
section("4. Slicing several dimensions: one entry per dimension, separated by commas")
# ':' alone means "all of this dimension". Dimensions you leave out at the end count as ':'.

show("x[:, 0]       # every block, position 0, all channels  -> (B, C)", x[:, 0])
show("x[:, -1]      # every block, the last position          -> (B, C)", x[:, -1])
show("x[:, -1, :]   # the same: the trailing ':' is optional", x[:, -1, :])
show("x[:, :, 0]    # every block, every position, channel 0  -> (B, T)", x[:, :, 0])
show("x[:, 1:, :2]  # positions 1 to end, channels 0 and 1    -> (B, 2, 2)", x[:, 1:, :2])
show("x[..., 0]     # '...' means 'all the dimensions before': the same as x[:, :, 0]", x[..., 0])

# ----------------------------------------------------------------------
section("5. 'dim' in functions: the dimension that gets used up")
# sum, mean, softmax, cat... take a dim. For sum and mean, that dimension disappears:
# the numbers along it are combined into one.

m = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])
show("m", m)
show("m.sum(dim=0)    # down the columns: one sum per column      -> (3,)", m.sum(dim=0))
show("m.sum(dim=1)    # along each row: one sum per row           -> (2,)", m.sum(dim=1))
show("m.sum(dim=-1)   # the last dimension: the same as dim=1 here", m.sum(dim=-1))
show("m.sum(dim=-1, keepdim=True)   # the dimension stays, with size 1 -> (2, 1)",
     m.sum(dim=-1, keepdim=True))
show("torch.softmax(m, dim=-1)   # softmax keeps the shape: each row now sums to 1",
     torch.softmax(m, dim=-1))
print("Rule of thumb: dim=-1 works 'within each row' (along the last dimension).\n")

# ----------------------------------------------------------------------
section("6. Changing the shape: view, transpose, cat")

show("x.view(6, 4)          # (B*T, C): the same 24 numbers, 2 blocks of 3 rows -> 6 rows",
     x.view(6, 4))
show("x.view(2*3*4)         # one long vector", x.view(24))
show("x.transpose(-2, -1)   # swap the last two dimensions: (B, T, C) -> (B, C, T)",
     x.transpose(-2, -1))
a = torch.zeros(2, 3, 2, dtype=torch.long)
b = torch.ones(2, 3, 2, dtype=torch.long)
show("torch.cat([a, b], dim=-1)   # (2,3,2) + (2,3,2) side by side in the last dim -> (2,3,4)",
     torch.cat([a, b], dim=-1))
show("torch.cat([a, b], dim=1)    # stacked along T instead -> (2,6,2)",
     torch.cat([a, b], dim=1))
show("torch.stack([v[:3], v[3:6]])   # stack ADDS a new dimension: two (3,) -> (2, 3)",
     torch.stack([v[:3], v[3:6]]))

# ----------------------------------------------------------------------
section("7. Broadcasting: a missing dimension is repeated")
# Shapes are lined up from the right. A missing or size-1 dimension is repeated to match.

pos = torch.tensor([[100, 100, 100, 100],
                    [200, 200, 200, 200],
                    [300, 300, 300, 300]])
show("pos               # (T, C) = (3, 4), like pos_emb", pos)
show("x + pos           # (2,3,4) + (3,4): pos is added to every block", x + pos)

# ----------------------------------------------------------------------
section("8. Indexing with a tensor of ids (what nn.Embedding does)")

table = torch.tensor([[0.0, 0.0],
                      [1.1, 1.2],
                      [2.1, 2.2],
                      [3.1, 3.2]])          # 4 rows (ids 0 to 3), 2 numbers each
ids = torch.tensor([[3, 1, 1],
                    [0, 2, 3]])             # (B, T) = (2, 3)
show("table             # (4, 2): one row per id", table)
show("ids               # (2, 3)", ids)
show("table[ids]        # each id replaced by its row -> (2, 3, 2)", table[ids])

# ----------------------------------------------------------------------
section("9. Masks: True/False tensors")

w = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]])
tril = torch.tril(torch.ones(3, 3))
show("tril              # 1 on and below the diagonal", tril)
show("tril == 0         # a boolean mask: True above the diagonal", tril == 0)
show("w.masked_fill(tril == 0, float('-inf'))   # replace where the mask is True",
     w.masked_fill(tril == 0, float('-inf')))
show("torch.where(w > 4, w, 0)   # element by element: w where True, else 0",
     torch.where(w > 4, w, torch.tensor(0.)))
show("w[w > 4]          # boolean indexing: the selected values, flattened -> (5,)", w[w > 4])
show("(w > 4).any(dim=-1)   # one True/False per row: is any value in the row > 4?",
     (w > 4).any(dim=-1))
show("(w > 4).all()     # one True/False for the whole tensor", (w > 4).all())

# ----------------------------------------------------------------------
section("10. Slices share memory with the original (a view); clone() makes a copy")

orig = torch.arange(6)
view = orig[2:4]
view[0] = 99
show("orig after view[0] = 99   # the original changed too", orig)
orig = torch.arange(6)
copy = orig[2:4].clone()
copy[0] = 99
show("orig after copy[0] = 99   # unchanged: clone() made a separate copy", orig)

# ----------------------------------------------------------------------
section("11. The slices in gpt.py")

data = torch.arange(20)                     # stands in for the encoded text
start = 5
show("data[start:start+8]       # get_batch: x, 8 characters from start", data[start:start + 8])
show("data[start+1:start+9]     # get_batch: y, the same shifted by one", data[start + 1:start + 9])

xb = torch.arange(12).view(2, 6)            # (B, T) = (2, 6)
show("xb                        # (B, T)", xb)
show("xb[:, 1:]                 # every block, drop the first position", xb[:, 1:])
show("xb[:, :-1]                # every block, drop the last position", xb[:, :-1])
show("xb[:, -4:]                # generate's crop idx[:, -BLOCK_SIZE:], with BLOCK_SIZE = 4",
     xb[:, -4:])
show("xb[:, -1]                 # the last character of every block -> (B,)", xb[:, -1])
show("xb[:, -1][1]              # the last column, then its element 1: ONE block's last character",
     xb[:, -1][1])

logits = torch.arange(2 * 3 * 5).view(2, 3, 5).float()   # (B, T, vocab) with vocab = 5
show("logits[:, -1, :]          # generate: the last time step's scores -> (B, vocab)",
     logits[:, -1, :])
show("logits[:, :-1]            # the no-peeking test: every position except the last",
     logits[:, :-1])
show("logits.view(2*3, 5)       # the loss: (B*T, vocab), one row per prediction",
     logits.view(6, 5))

big_tril = torch.tril(torch.ones(4, 4))
show("big_tril[:2, :2]          # Head: tril[:T, :T] when T = 2 (in generate)", big_tril[:2, :2])
