'''
Learning Rate Scheduler
'''

import math

import torch
from torch import nn
from torch.optim import lr_scheduler

from dl_utils.d2l.data_fashion import load_data_fashion_mnist
from dl_utils.evaluation.supervised import accuracy, evaluate_accuracy_gpu
from dl_utils.plot._backend import pyplot as plt
from dl_utils.plot.figures import Animator, plot
from dl_utils.runtime.devices import try_gpu
from dl_utils.training.metrics import Accumulator


def net_fn():
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
        nn.Linear(120, 84), nn.ReLU(),
        nn.Linear(84, 10))

    return model

loss = nn.CrossEntropyLoss()
device = try_gpu()

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

# Almost identical to train_ch6 defined in the LeNet section of the CNN chapter
def train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
          scheduler=None):
    net.to(device)
    animator = Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        metric = Accumulator(3)  # train_loss,train_acc,num_examples
        for i, (X, y) in enumerate(train_iter):
            net.train()
            trainer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            trainer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_loss = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % 50 == 0:
                animator.add(epoch + i / len(train_iter),
                             (train_loss, train_acc, None))

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))

        if scheduler:
            if scheduler.__module__ == lr_scheduler.__name__:
                # Using PyTorch In-Built scheduler
                scheduler.step()
            else:
                # Using custom defined scheduler
                for param_group in trainer.param_groups:
                    param_group['lr'] = scheduler(epoch)

    print(f'train loss {train_loss:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')

lr, num_epochs = 0.3, 30
net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device)

lr = 0.1
trainer.param_groups[0]["lr"] = lr
print(f'learning rate is now {trainer.param_groups[0]["lr"]:.2f}')

class SquareRootScheduler:
    def __init__(self, lr=0.1):
        self.lr = lr

    # Make the instance of the class callable like a function
    def __call__(self, num_update):
        return self.lr * pow(num_update + 1.0, -0.5)

scheduler = SquareRootScheduler(lr=0.1)
plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr)
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
#       scheduler)

class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
plot(torch.arange(50), [scheduler(t) for t in range(50)])

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.5)
scheduler = lr_scheduler.MultiStepLR(trainer, milestones=[15, 30], gamma=0.5)

def get_lr(trainer, scheduler):
    lr = scheduler.get_last_lr()[0]
    trainer.step()
    scheduler.step()
    return lr

plot(torch.arange(num_epochs), [get_lr(trainer, scheduler)
                                  for t in range(num_epochs)])

# train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
#       scheduler)

class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr          # original base learning rate before decay
        self.max_update = max_update         # total number of training steps
        self.final_lr = final_lr             # minimum learning rate at the end of cosine decay
        self.warmup_steps = warmup_steps     # number of steps for linear warmup phase
        self.warmup_begin_lr = warmup_begin_lr  # initial learning rate at the start of warmup
        self.max_steps = self.max_update - self.warmup_steps  # steps reserved for cosine decay

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
# train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
#       scheduler)

scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])

net = net_fn()
trainer = torch.optim.SGD(net.parameters(), lr=0.3)
train(net, train_iter, test_iter, num_epochs, loss, trainer, device,
      scheduler)

# plt.ioff()
# plt.show()
