from __future__ import annotations

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import chex
import gymnasium
import mujoco
import numpy as np
from flax import struct
from gymnasium.core import RenderFrame
from mujoco import mjx

import moojoco.environment.mjx_spaces as mjx_spaces
from moojoco.environment.renderer import MujocoRenderer
from moojoco.mjcf.arena import MJCFArena
from moojoco.mjcf.morphology import MJCFMorphology


class MuJoCoEnvironmentConfiguration:
    PHYSICS_TIMESTEP = 0.002

    def __init__(
        self,
        time_scale: float = 1.0,
        num_physics_steps_per_control_step: int = 1,
        simulation_time: float = 10,
        camera_ids: List[int] | None = None,
        render_size: Tuple[int, int] | None = (240, 320),
        render_mode: Optional[str] = None,
        solver_iterations: int | None = None,
        solver_ls_iterations: int | None = None,
        disable_eulerdamp: bool | None = None,
    ) -> None:
        self.time_scale = time_scale
        self.num_physics_steps_per_control_step = num_physics_steps_per_control_step
        self.simulation_time = simulation_time
        self.camera_ids = camera_ids or [0]
        self.render_size = render_size
        self.render_mode = render_mode
        self.solver_iterations = solver_iterations
        self.solver_ls_iterations = solver_ls_iterations
        self.disable_eulerdamp = disable_eulerdamp

    @property
    def control_timestep(self) -> float:
        return self.num_physics_steps_per_control_step * self.physics_timestep

    @property
    def physics_timestep(self) -> float:
        return self.PHYSICS_TIMESTEP * self.time_scale

    @property
    def total_num_control_steps(self) -> int:
        return int(np.ceil(self.simulation_time / self.control_timestep))

    @property
    def total_num_physics_steps(self) -> int:
        return int(np.ceil(self.simulation_time / self.physics_timestep))


ModelType = Union[mujoco.MjModel, mjx.Model]
DataType = Union[mujoco.MjData, mjx.Data]
SpaceType = Union[gymnasium.spaces.Space, mjx_spaces.Space]
BoxSpaceType = Union[gymnasium.spaces.Box, mjx_spaces.Box]
DictSpaceType = Union[gymnasium.spaces.Dict, mjx_spaces.Dict]


class BaseObservable:
    def __init__(
        self,
        name: str,
        low: chex.Array,
        high: chex.Array,
        retriever: Callable[[BaseEnvState], chex.Array],
    ) -> None:
        self.name = name
        self.low = low
        self.high = high
        self.retriever = retriever

    def __call__(self, state: BaseEnvState) -> chex.Array:
        return self.retriever(state)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.low.shape


@struct.dataclass
class BaseEnvState(abc.ABC):
    mj_model: mujoco.MjModel = struct.field(pytree_node=False)
    mj_data: mujoco.MjData = struct.field(pytree_node=False)
    observations: Dict[str, chex.Array]
    reward: chex.Array
    terminated: chex.Array
    truncated: chex.Array
    info: Dict[str, Any]
    rng: np.random.RandomState | chex.PRNGKey


class BaseEnvironment(abc.ABC):
    def __init__(self, configuration: MuJoCoEnvironmentConfiguration) -> None:
        self._configuration = configuration

    @property
    def environment_configuration(self) -> MuJoCoEnvironmentConfiguration:
        return self._configuration

    @property
    @abc.abstractmethod
    def action_space(self) -> SpaceType:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def actuators(self) -> List[str]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def observation_space(self) -> SpaceType:
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self, state: BaseEnvState, action: chex.Array, *args, **kwargs
    ) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
        self, rng: np.random.RandomState | chex.PRNGKey, *args, **kwargs
    ) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def render(self, state: BaseEnvState) -> List[RenderFrame] | None:
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class BaseMuJoCoEnvironment(BaseEnvironment, abc.ABC):
    metadata = {"render_modes": []}

    def __init__(
        self,
        mjcf_str: str,
        mjcf_assets: Dict[str, Any],
        configuration: MuJoCoEnvironmentConfiguration,
    ) -> None:
        super().__init__(configuration=configuration)
        self._mjcf_str = mjcf_str
        self._mjcf_assets = mjcf_assets

        self._renderers: Dict[int, Union[MujocoRenderer, mujoco.Renderer]] = dict()

        self._mj_model, self._mj_data = self._initialize_mj_model_and_data()

        assert (
            configuration.render_mode is None
            or self.environment_configuration.render_mode
            in self.metadata["render_modes"]
        ), (
            f"Unsupported render mode: '{self.environment_configuration.render_mode}'. Must be one "
            f"of {self.metadata['render_modes']}"
        )

        self._observables: Optional[List[BaseObservable]] = self._create_observables()
        self._action_space: Optional[BoxSpaceType] = self._create_action_space()
        self._observation_space: Optional[DictSpaceType] = (
            self._create_observation_space()
        )

    @classmethod
    def from_morphology_and_arena(
        cls,
        morphology: MJCFMorphology,
        arena: MJCFArena,
        configuration: MuJoCoEnvironmentConfiguration,
    ) -> BaseMuJoCoEnvironment:
        arena.attach(other=morphology, free_joint=True)
        mjcf_str, mjcf_assets = arena.get_mjcf_str(), arena.get_mjcf_assets()
        return cls(
            mjcf_str=mjcf_str, mjcf_assets=mjcf_assets, configuration=configuration
        )

    @property
    def frozen_mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def frozen_mj_data(self) -> mujoco.MjData:
        return self._mj_data

    @property
    def actuators(self) -> List[str]:
        return [
            self.frozen_mj_model.actuator(i).name
            for i in range(self.frozen_mj_model.nu)
        ]

    @property
    def action_space(self) -> SpaceType:
        return self._action_space

    @property
    def observation_space(self) -> SpaceType:
        return self._observation_space

    def _initialize_mj_model_and_data(self) -> Tuple[mujoco.MjModel, mujoco.MjData]:
        mj_model = mujoco.MjModel.from_xml_string(
            xml=self._mjcf_str, assets=self._mjcf_assets
        )
        mj_model.vis.global_.offheight = self.environment_configuration.render_size[0]
        mj_model.vis.global_.offwidth = self.environment_configuration.render_size[1]
        mj_model.opt.timestep = self.environment_configuration.physics_timestep
        if self.environment_configuration.solver_iterations:
            mj_model.opt.iterations = self.environment_configuration.solver_iterations
        if self.environment_configuration.solver_ls_iterations:
            mj_model.opt.ls_iterations = (
                self.environment_configuration.solver_ls_iterations
            )
        if self.environment_configuration.disable_eulerdamp:
            mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
        mj_data = mujoco.MjData(mj_model)
        return mj_model, mj_data

    def step(
        self, state: BaseEnvState, action: chex.Array, *args, **kwargs
    ) -> BaseEnvState:
        previous_state = state
        state = self._update_simulation(state=state, ctrl=action)

        state = self._update_observations(state=state)
        state = self._update_reward(state=state, previous_state=previous_state)
        state = self._update_terminated(state=state)
        state = self._update_truncated(state=state)
        state = self._update_info(state=state)

        return state

    def get_renderer(
        self,
        identifier: int,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        state: BaseEnvState,
    ) -> Union[MujocoRenderer, mujoco.Renderer]:
        if identifier not in self._renderers:
            if self.environment_configuration.render_mode == "human":
                renderer = MujocoRenderer(
                    model=model, data=data, default_cam_config=None
                )
            else:
                renderer = mujoco.Renderer(
                    model=model,
                    height=self.environment_configuration.render_size[0],
                    width=self.environment_configuration.render_size[1],
                )
            self._renderers[identifier] = renderer
        return self._renderers[identifier]

    def get_renderer_context(
        self, renderer: Union[MujocoRenderer, mujoco.Renderer]
    ) -> mujoco.MjrContext:
        try:
            # noinspection PyProtectedMember
            context = renderer._get_viewer(
                render_mode=self.environment_configuration.render_mode
            ).con
        except AttributeError:
            # noinspection PyProtectedMember
            context = renderer._mjr_context
        return context

    @abc.abstractmethod
    def _get_mj_models_and_datas_to_render(
        self, state: BaseEnvState
    ) -> Tuple[List[mujoco.MjModel], List[mujoco.MjData]]:
        raise NotImplementedError

    def render(self, state: BaseEnvState) -> List[RenderFrame] | None:
        camera_ids = self.environment_configuration.camera_ids or [-1]

        mj_models, mj_datas = self._get_mj_models_and_datas_to_render(state=state)

        frames = []
        for i, (model, data) in enumerate(zip(mj_models, mj_datas)):
            mujoco.mj_forward(m=model, d=data)
            renderer = self.get_renderer(
                identifier=i, model=model, data=data, state=state
            )

            if self.environment_configuration.render_mode == "human":
                viewer = renderer._get_viewer("human")
                viewer.model = model
                viewer.data = data

                return renderer.render(render_mode="human", camera_id=camera_ids[0])
            else:
                for camera_id in camera_ids:
                    renderer._model = model
                    renderer.update_scene(data=data, camera=camera_id)
                    frame = renderer.render()[:, :, ::-1]
                    frames.append(frame)

        return frames

    def _close_renderers(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()

    def close(self) -> None:
        self._close_renderers()
        del self._mj_model
        del self._mj_data
        del self._observation_space
        del self._action_space
        del self._observables

    @abc.abstractmethod
    def _create_observation_space(self) -> DictSpaceType:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_observations(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_action_space(self) -> BoxSpaceType:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_simulation(self, state: BaseEnvState, ctrl: chex.Array) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _prepare_reset(self) -> Tuple[Tuple[ModelType, DataType], ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def _finish_reset(
        self,
        models_and_datas: Tuple[Tuple[ModelType, DataType], ...],
        rng: np.random.RandomState | chex.PRNGKey,
    ) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(
        self, rng: np.random.RandomState | chex.PRNGKey, *args, **kwargs
    ) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_observables(self) -> List[BaseObservable]:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_reward(
        self, state: BaseEnvState, previous_state: BaseEnvState
    ) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_terminated(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_truncated(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError

    @abc.abstractmethod
    def _update_info(self, state: BaseEnvState) -> BaseEnvState:
        raise NotImplementedError
