from __future__ import annotations

import abc
import copy
from abc import ABC
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Tuple

import gymnasium
import mujoco
import numpy as np
from flax import struct
from gymnasium.core import ActType, RenderFrame
from gymnasium.vector.utils import batch_space

from moojoco.environment.base import (
    BaseEnvState,
    BaseEnvironment,
    BaseMuJoCoEnvironment,
    BaseObservable,
    MuJoCoEnvironmentConfiguration,
)
from moojoco.mjcf.arena import MJCFArena
from moojoco.mjcf.morphology import MJCFMorphology


@struct.dataclass
class MJCEnvState(BaseEnvState):
    observations: Dict[str, np.ndarray]
    rng: np.random.RandomState


class MJCObservable(BaseObservable):
    def __init__(
        self,
        name: str,
        low: np.ndarray,
        high: np.ndarray,
        retriever: Callable[[MJCEnvState], np.ndarray],
    ) -> None:
        super().__init__(name=name, low=low, high=high, retriever=retriever)

    def __call__(self, state: MJCEnvState) -> np.ndarray:
        return super().__call__(state=state)


class MJCEnv(BaseMuJoCoEnvironment, ABC):
    def __init__(
        self,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
        configuration: MuJoCoEnvironmentConfiguration,
    ) -> None:
        super().__init__(
            mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration
        )

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFMorphology,
        arena: MJCFArena,
        configuration: MuJoCoEnvironmentConfiguration,
    ) -> MJCEnv:
        return super().from_morphology_and_arena(
            morphology=morphology, arena=arena, configuration=configuration
        )

    def _get_mj_models_and_datas_to_render(
        self, state: MJCEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        return [state.mj_model], [state.mj_data]

    @property
    def observables(self) -> List[MJCObservable]:
        return self._observables

    def _create_observation_space(self) -> gymnasium.spaces.Dict:
        observation_space = dict()
        for observable in self.observables:
            observation_space[observable.name] = gymnasium.spaces.Box(
                low=observable.low, high=observable.high, shape=observable.shape
            )
        return gymnasium.spaces.Dict(observation_space)

    def _create_action_space(self) -> gymnasium.Space:
        bounds = self.frozen_mj_model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)
        return action_space

    def _update_observations(self, state: MJCEnvState) -> MJCEnvState:
        observations = dict()
        for observable in self.observables:
            observations[observable.name] = observable(state=state)
        # noinspection PyUnresolvedReferences
        return state.replace(observations=observations)

    def _update_simulation(self, state: MJCEnvState, ctrl: ActType) -> MJCEnvState:
        state.mj_data.ctrl[:] = ctrl
        mujoco.mj_step(
            m=state.mj_model,
            d=state.mj_data,
            nstep=self.environment_configuration.num_physics_steps_per_control_step,
        )
        return state

    def _prepare_reset(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        mujoco.mj_resetData(self.frozen_mj_model, self.frozen_mj_data)
        return self.frozen_mj_model, self.frozen_mj_data

    def _finish_reset(
        self,
        models_and_datas: Tuple[mujoco.MjModel, mujoco.MjData],
        rng: np.random.RandomState,
    ) -> MJCEnvState:
        mj_model, mj_data = models_and_datas
        mujoco.mj_forward(m=mj_model, d=mj_data)
        state = MJCEnvState(
            mj_model=mj_model,
            mj_data=mj_data,
            observations={},
            reward=0,
            terminated=False,
            truncated=False,
            info={},
            rng=rng,
        )
        state = self._update_observations(state=state)
        state = self._update_info(state=state)
        return state

    @abc.abstractmethod
    def _create_observables(self) -> List[MJCObservable]:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self, rng: np.random.RandomState, *args, **kwargs) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_reward(
        self, state: MJCEnvState, previous_state: MJCEnvState
    ) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_terminated(self, state: MJCEnvState) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_truncated(self, state: MJCEnvState) -> MJCEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_info(self, state: MJCEnvState) -> MJCEnvState:
        raise NotImplementedError


@struct.dataclass
class VectorMJCEnvState(MJCEnvState):
    mj_model: List[mujoco.MjModel] = struct.field(pytree_node=False)
    mj_data: List[mujoco.MjData] = struct.field(pytree_node=False)
    observations: Dict[str, np.ndarray]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: Dict[str, Any]
    rng: List[np.random.RandomState]


class ThreadedVectorMJCEnvWrapper(BaseEnvironment):
    def __init__(
        self, create_env_fn: Callable[[], BaseEnvironment], num_environments: int
    ) -> None:
        self._create_env_fn = create_env_fn
        self._num_environments = num_environments
        dummy_env = create_env_fn()
        self._single_action_space = dummy_env.action_space
        self._action_space = batch_space(
            self._single_action_space, self._num_environments
        )
        self._observation_space = batch_space(
            dummy_env.observation_space, self._num_environments
        )
        self._actuators = dummy_env.actuators
        super().__init__(configuration=dummy_env.environment_configuration)
        dummy_env.close()

        self._pool = ThreadPoolExecutor(max_workers=num_environments)
        self._envs = self._create_envs()
        self._states = None

    def _create_envs(self) -> List[MJCEnv]:
        environments = list(
            self._pool.map(
                lambda _: self._create_env_fn(), range(self._num_environments)
            )
        )
        return environments

    @property
    def _merged_states(self) -> VectorMJCEnvState:
        mj_models = []
        mj_datas = []
        observations = defaultdict(list)
        reward = []
        terminated = []
        truncated = []
        info = defaultdict(list)
        rng = []
        for env_id, state in enumerate(self._states):
            mj_models.append(state.mj_model)
            mj_datas.append(state.mj_data)
            for k, o in state.observations.items():
                observations[k].append(o)
            reward.append(state.reward)
            terminated.append(state.terminated)
            truncated.append(state.truncated)
            for k, o in state.info.items():
                info[k].append(o)
            rng.append(state.rng)

        observations = {k: np.stack(v) for k, v in observations.items()}
        info = {k: np.array(v) for k, v in info.items()}

        return VectorMJCEnvState(
            mj_model=mj_models,
            mj_data=mj_datas,
            observations=observations,
            reward=np.array(reward),
            terminated=np.array(terminated),
            truncated=np.array(truncated),
            info=info,
            rng=rng,
        )

    @property
    def single_action_space(self) -> gymnasium.spaces.Space:
        return self._single_action_space

    @property
    def action_space(self) -> gymnasium.spaces.Space:
        return self._action_space

    @property
    def actuators(self) -> List[str]:
        return self._actuators

    @property
    def observation_space(self) -> gymnasium.spaces.Space:
        return self._observation_space

    def step(
        self, state: VectorMJCEnvState, action: np.ndarray, *args, **kwargs
    ) -> VectorMJCEnvState:
        self._states = list(
            self._pool.map(
                lambda env, ste, act: env.step(ste, act, *args, **kwargs),
                self._envs,
                self._states,
                action,
            )
        )
        return self._merged_states

    def reset(
        self, rng: List[np.random.RandomState], *args, **kwargs
    ) -> VectorMJCEnvState:
        self._states = list(
            self._pool.map(
                lambda env, sub_rng: env.reset(sub_rng, *args, **kwargs),
                self._envs,
                rng,
            )
        )

        return self._merged_states

    def render(self, state: VectorMJCEnvState) -> List[RenderFrame] | None:
        if self.environment_configuration.render_mode == "human":
            # Only render first env; Has to be in main thread
            return self._envs[0].render(state=self._states[0])

        frames = []
        for env, state in zip(self._envs, self._states):
            frames += env.render(state=state)
        return frames

    def close(self) -> None:
        self._pool.shutdown()
        for env in self._envs:
            env.close()
