import abc
from typing import List, Tuple

import chex
import numpy as np
from gymnasium.core import RenderFrame

from moojoco.environment.base import BaseEnvState, BaseEnvironment, SpaceType


class PostInitCaller(type):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post__init__()
        return obj


class CombinedMeta(abc.ABCMeta, PostInitCaller):
    pass


class EnvironmentWrapper(BaseEnvironment, metaclass=CombinedMeta):
    def __init__(self, env: BaseEnvironment) -> None:
        super().__init__(configuration=env.environment_configuration)
        self._env = env

        self._observation_space: SpaceType | None = None
        self._action_space: SpaceType | None = None

    def __post__init__(self) -> None:
        # Make sure observation space and action space are initialised after creation
        # noinspection PyStatementEffect
        self.observation_space
        # noinspection PyStatementEffect
        self.action_space

    @property
    def action_space(self) -> SpaceType:
        if self._action_space is not None:
            return self._action_space
        return self._env.action_space

    @property
    def actuators(self) -> List[str]:
        return self._env.actuators

    @property
    def observation_space(self) -> SpaceType:
        if self._observation_space is not None:
            return self._observation_space
        return self._env.observation_space

    def step(
        self, state: BaseEnvState, action: chex.Array, *args, **kwargs
    ) -> BaseEnvState:
        return self._env.step(state, action, *args, **kwargs)

    def reset(
        self, rng: np.random.RandomState | chex.PRNGKey, *args, **kwargs
    ) -> BaseEnvState:
        return self._env.reset(rng, *args, **kwargs)

    def render(self, state: BaseEnvState) -> List[RenderFrame] | None:
        return self._env.render(state=state)

    def close(self) -> None:
        return self._env.close()


class TransformObservationEnvWrapper(EnvironmentWrapper, abc.ABC):
    def __init__(self, env: BaseEnvironment) -> None:
        super().__init__(env=env)

    @property
    @abc.abstractmethod
    def observation_space(self) -> SpaceType:
        raise NotImplementedError

    @abc.abstractmethod
    def _transform_observations(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError

    def step(
        self, state: BaseEnvState, action: chex.Array, *args, **kwargs
    ) -> BaseEnvState:
        state = self._env.step(state, action, *args, **kwargs)
        state = self._transform_observations(state=state)
        return state

    def reset(
        self, rng: np.random.RandomState | chex.PRNGKey, *args, **kwargs
    ) -> BaseEnvState:
        state = self._env.reset(rng, *args, **kwargs)
        state = self._transform_observations(state=state)
        return state


class TransformActionEnvWrapper(EnvironmentWrapper, abc.ABC):
    def __init__(self, env: BaseEnvironment) -> None:
        super().__init__(env=env)

    @property
    @abc.abstractmethod
    def action_space(self) -> SpaceType:
        raise NotImplementedError

    @abc.abstractmethod
    def _transform_action(
        self, action: chex.Array, state: BaseEnvState
    ) -> Tuple[chex.Array, BaseEnvState]:
        raise NotImplementedError

    def step(
        self, state: BaseEnvState, action: chex.Array, *args, **kwargs
    ) -> BaseEnvState:
        action, state = self._transform_action(action=action, state=state)
        state = self._env.step(state, action, *args, **kwargs)
        return state
