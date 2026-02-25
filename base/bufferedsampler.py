from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Optional
from utils.types import NPFArray, f32


class RandomVariate(Protocol):
    def rvs(self, size: int) -> NPFArray: ...


class BufferedSampler:
    def __init__(self, source: RandomVariate, name: str, size: int = 100_000):
        self.source = source
        self.name = name
        self.size = size
        self.cur_index = 0
        self.cur_arr = self.source.rvs(self.size)

    def get(self):
        if self.cur_index >= self.size:
            self.cur_arr = self.source.rvs(self.size)
            self.cur_index = 0
        v = self.cur_arr[self.cur_index]
        self.cur_index += 1
        return v

    def get_full(self) -> NPFArray:
        return self.source.rvs(self.size)
