from dl_utils.training.timing import Timer

class Benchmark:
    """For measuring running time"""
    def __init__(self, description='Done'):
        self.description = description
        self.timer = None

    def __enter__(self):
        # __enter__ is a dunder method that implements the context manager protocol;
        # it is automatically called when execution enters a `with` block,
        # allowing setup logic (here: starting the timer) to run before the block body
        self.timer = Timer()
        return self

    def __exit__(self, *args):
        # __exit__ is a dunder method that implements the context manager protocol;
        # it is automatically called when execution exits a `with` block,
        # allowing cleanup logic (here: stopping the timer and printing the result) to run after the block body
        assert self.timer is not None
        print(f'{self.description}: {self.timer.stop():.4f} sec')
