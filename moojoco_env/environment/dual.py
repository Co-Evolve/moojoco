from __future__ import annotations

from typing import List

import chex
import numpy as np
from gymnasium.core import RenderFrame

from moojoco_env.mjcf.arena import MJCFArena
from moojoco_env.environment.base import (
    BaseEnvState,
    BaseEnvironment,
    MuJoCoEnvironmentConfiguration,
    SpaceType,
)
from moojoco_env.environment.mjc_env import MJCEnv
from moojoco_env.environment.mjx_env import MJXEnv
from moojoco_env.mjcf.morphology import MJCFMorphology


class DualMuJoCoEnvironment(BaseEnvironment):
    MJC_ENV_CLASS: type[MJCEnv]
    MJX_ENV_CLASS: type[MJXEnv]

    def __init__(self, env: MJCEnv | MJXEnv, backend: str) -> None:
        super().__init__(configuration=env.environment_configuration)
        self.backend = backend
        self._env = env

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFMorphology,
        arena: MJCFArena,
        configuration: MuJoCoEnvironmentConfiguration,
        backend: str,
    ) -> DualMuJoCoEnvironment:
        assert backend in [
            "MJC",
            "MJX",
        ], f"Backend must either be 'MJC' or 'MJX'. {backend} was given."
        if backend == "MJC":
            env_class = cls.MJC_ENV_CLASS
        else:
            env_class = cls.MJX_ENV_CLASS
        env = env_class.from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )
        return cls(env=env, backend=backend)

    @property
    def action_space(self) -> SpaceType:
        return self._env.action_space

    @property
    def actuators(self) -> List[str]:
        return self._env.actuators

    @property
    def observation_space(self) -> SpaceType:
        return self._env.observation_space

    def step(self, state: BaseEnvState, action: chex.Array) -> BaseEnvState:
        return self._env.step(state=state, action=action)

    def reset(self, rng: np.random.RandomState | chex.PRNGKey) -> BaseEnvState:
        return self._env.reset(rng=rng)

    def render(self, state: BaseEnvState) -> List[RenderFrame] | None:
        return self._env.render(state=state)

    def close(self) -> None:
        return self._env.close()
