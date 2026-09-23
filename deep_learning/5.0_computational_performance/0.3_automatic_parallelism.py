"""
Automatic Parallelism
"""

import torch
from dl_utils.d2l.benchmark import Benchmark
from dl_utils.runtime.devices import try_gpu

def run(x):
    # Matrix multiply x by itself 50 times
    return [x.mm(x) for _ in range(50)]

def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]


def main():
    device = try_gpu(0)
    print(device)

    # Generate random tensors uniformly distributed in [0, 1)
    x_gpu = torch.rand(size=(4000, 4000), device=device)

    run(x_gpu)  # Warm up device
    # block CPU until GPU finishes all pending ops on this device
    torch.cuda.synchronize(device)

    with Benchmark('GPU0 time'):
        run(x_gpu)
        torch.cuda.synchronize(device)

    # non_blocking=True only reduces CPU wait time. All ops still queue
    # on the same default stream, so GPU execution is serial:
    #   matmul0→...→matmul49 → copy0→...→copy49
    # True compute/copy overlap requires multiple CUDA streams.

    with Benchmark('Running on GPU0'):
        y = run(x_gpu)
        torch.cuda.synchronize()

    with Benchmark('Copy to CPU'):
        y_cpu = copy_to_cpu(y)
        torch.cuda.synchronize()

    with Benchmark('Running on GPU0 and copying to CPU'):
        y = run(x_gpu)
        y_cpu = copy_to_cpu(y, True)
        torch.cuda.synchronize()


if __name__ == '__main__':
    main()
