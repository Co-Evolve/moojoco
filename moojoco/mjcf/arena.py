from abc import ABC

from moojoco.mjcf.component import MJCFRootComponent


class ArenaConfiguration(ABC):
    def __init__(self, name: str) -> None:
        self.name = name


class MJCFArena(MJCFRootComponent, ABC):
    def __init__(self, configuration: ArenaConfiguration, *args, **kwargs) -> None:
        self._configuration = configuration
        super().__init__(name=configuration.name, *args, **kwargs)

    @property
    def arena_configuration(self) -> ArenaConfiguration:
        return self._configuration

    def _build(self, *args, **kwargs) -> None:
        raise NotImplementedError
