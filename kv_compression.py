import torch

class KVCompressor:
    def __init__(self, enabled: bool = False, window_size: int = None, sink_size: int = 0):
        self.enabled = enabled
        self.window_size = window_size
        self.sink_size = sink_size
