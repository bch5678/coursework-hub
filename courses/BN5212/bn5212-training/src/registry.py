"""Name -> builder registries so experiments stay declarative.

Adding a new fusion strategy means adding one module that registers itself; no
other file in the framework has to change. This is the plug-in contract the
project plan asks for in section 9.
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._builders: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorate(builder: Callable[..., T]) -> Callable[..., T]:
            if name in self._builders:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._builders[name] = builder
            return builder

        return decorate

    def build(self, name: str, *args, **kwargs) -> T:
        if name not in self._builders:
            raise KeyError(
                f"Unknown {self.kind} {name!r}. Available: {sorted(self._builders)}"
            )
        return self._builders[name](*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._builders)

    def __contains__(self, name: object) -> bool:
        return name in self._builders


IMAGE_ENCODERS: Registry = Registry("image encoder")
CLINICAL_ENCODERS: Registry = Registry("clinical encoder")
FUSIONS: Registry = Registry("fusion module")
