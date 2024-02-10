from __future__ import annotations

from abc import ABC
from typing import List, Union

import numpy as np
from dm_control import mjcf
from fprs.robot import Morphology
from fprs.specification import MorphologySpecification

from moojoco.mjcf.component import MJCFRootComponent, MJCFSubComponent


class MJCFMorphology(Morphology, MJCFRootComponent, ABC):
    def __init__(
        self,
        specification: MorphologySpecification,
        name: str = "morphology",
        *args,
        **kwargs,
    ) -> None:
        Morphology.__init__(self, specification=specification)
        MJCFRootComponent.__init__(self, name=name, *args, **kwargs)

    @property
    def actuators(self) -> List[mjcf.Element]:
        return self.mjcf_model.find_all("actuator")

    @property
    def sensors(self) -> List[mjcf.Element]:
        return self.mjcf_model.find_all("sensor")

    def _build(self, *args, **kwargs) -> None:
        raise NotImplementedError


class MJCFMorphologyPart(MJCFSubComponent, ABC):
    def __init__(
        self,
        parent: Union[MJCFMorphology, MJCFMorphologyPart],
        name: str,
        position: np.array,
        euler: np.array,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            parent=parent, name=name, position=position, euler=euler, *args, **kwargs
        )

    @property
    def parent(self) -> Union[MJCFMorphology, MJCFMorphologyPart]:
        return self._parent

    @property
    def morphology_specification(self) -> MorphologySpecification:
        return self.parent.morphology_specification
