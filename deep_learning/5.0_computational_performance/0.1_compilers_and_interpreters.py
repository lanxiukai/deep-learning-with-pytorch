"""
Compilers and Interpreters
"""

def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))

def add_():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_():
    return add_() + fancy_func_() + 'print(fancy_func(1, 2, 3, 4))'

prog = evoke_()
print(prog)
# compile() converts the source code string into a bytecode object (code object):
#   - arg 1 prog  : the source code string to compile
#   - arg 2 ''    : pseudo filename used only in error messages, left empty here
#   - arg 3 'exec': compilation mode — 'exec' handles a sequence of statements
#                   (as opposed to 'eval', which handles a single expression)
# Pre-compiling is more efficient than passing a raw string to exec() directly,
# since the bytecode object can be executed multiple times without recompilation.
y = compile(prog, '', 'exec')
# exec() runs the compiled bytecode object, equivalent to executing prog as Python code.
# This completes the full symbolic programming pipeline: build code as a string → compile → execute.
exec(y)

import subprocess

import torch
from torch import nn

from dl_utils.d2l.benchmark import Benchmark
from dl_utils.filesystem.project_root import infer_project_root

# Factory pattern for producing networks
def get_net():
    net = nn.Sequential(nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2))
    return net

x = torch.randn(size=(1, 512))
net = get_net()
print(net(x))

# torch.jit.script() compiles the model into TorchScript IR (intermediate representation):
#   - statically analyzes the source code of each submodule and method for type inference
#   - enables graph-level optimizations (e.g. operator fusion, constant folding)
#     that are impossible in eager mode, where ops are dispatched one at a time
#   - decouples the model from the CPython interpreter and GIL, making it
#     executable in pure C++ environments without a Python runtime
#   - allows the compiled model to be serialized and deployed via net.save()
# The output is functionally identical to the eager version above, but now runs
# through the compiled execution path instead of the Python interpreter.
net = torch.jit.script(net)
print(net(x))

net = get_net()
with Benchmark('without torchscript'):
    for i in range(100000):
        net(x)

net = torch.jit.script(net)
with Benchmark('with torchscript'):
    for i in range(100000): 
        net(x)

# net.save() serializes the TorchScript model into a self-contained binary archive (ZIP format),
# bundling the compiled IR (computation graph), all layer parameters (weights & biases),
# the module hierarchy, and type annotations — everything needed to reload and run
# the model via torch.jit.load() in Python or LibTorch in C++, with no Python class definitions required.
output_dir = infer_project_root() / 'output' / 'torchscript'
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / 'mlp'
net.save(str(model_path))
subprocess.run(['ls', '-lh', str(model_path)], check=True)
