"""Save & load a custom MLP with flops.Module.

Define your model as a flops.Module subclass; flopscope discovers its array
state automatically. `save` writes named numeric arrays + an inert JSON config;
`from_file` rebuilds the model from your class (code) + that config. No pickle,
ever — and loading the weights is free.

Run: uv run python examples/12_save_load_mlp.py
"""

import os
import tempfile

import flopscope as flops
import flopscope.numpy as fnp


class Linear(flops.Module):
    def __init__(self, n_in, n_out):
        self.W = fnp.random.randn(n_out, n_in) * fnp.sqrt(2.0 / n_in)
        self.b = fnp.zeros(n_out)

    def __call__(self, x):
        return fnp.maximum(fnp.einsum("oi,i->o", self.W, x) + self.b, 0.0)


class MLP(flops.Module):
    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.layers = [Linear(a, b) for a, b in zip(sizes, sizes[1:], strict=False)]

    def config(self):
        return {"sizes": self.sizes}

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


path = os.path.join(tempfile.gettempdir(), "flopscope_mlp.npz")

with flops.BudgetContext(flop_budget=10_000_000) as budget:
    mlp = MLP([8, 16, 4])
    x = fnp.random.randn(8)
    before = mlp(x)

    mlp.save(path)  # -> layers.0.W, layers.0.b, ... + __meta__
    restored = MLP.from_file(path)  # class from code, weights+config from file

    after = restored(x)
    print("restored sizes:", restored.sizes)
    print("outputs identical after save/load:", before.tolist() == after.tolist())
    print(budget.summary())
