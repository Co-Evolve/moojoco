from __future__ import annotations

import abc
from abc import ABC
from typing import Dict, Optional, Union

import numpy as np
from dm_control import mjcf
from dm_control.mjcf import export_with_assets
from dm_control.mjcf.element import _AttachmentFrame
from scipy.spatial.transform import Rotation


class MJCFRootComponent(ABC):
    def __init__(self, name: str, *args, **kwargs) -> None:
        self._name = name
        self._parent = None
        self._mjcf_model = mjcf.RootElement(model=name)
        self._build(*args, **kwargs)

    @property
    def base_name(self) -> str:
        return self._name

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._mjcf_model

    @property
    def mjcf_body(self) -> mjcf.Element:
        return self.mjcf_model.worldbody

    def attach(
        self,
        other: MJCFRootComponent,
        position: Optional[np.ndarray] = None,
        euler: Optional[np.ndarray] = None,
        free_joint: bool = False,
    ) -> _AttachmentFrame:
        attachment_site = self.mjcf_body.add(
            "site",
            name=f"{self.base_name}_attachment_{other.base_name}",
            pos=position,
            euler=euler,
        )
        other._parent = self

        frame = attachment_site.attach(other.mjcf_model)
        if free_joint:
            frame.add("freejoint", name=f"freejoint")
        return frame

    def detach(self) -> None:
        self.mjcf_model.detach()

    def export_to_xml_with_assets(self, output_directory: str = "./mjcf") -> None:
        export_with_assets(mjcf_model=self.mjcf_model, out_dir=output_directory)

    def get_mjcf_str(self) -> str:
        return self.mjcf_model.to_xml_string()

    def get_mjcf_assets(self) -> Dict:
        return self.mjcf_model.get_assets()

    @property
    def world_coordinates(self) -> np.ndarray:
        return np.zeros(3)

    @property
    def coordinate_frame_in_root_frame(self) -> (np.ndarray, np.ndarray):
        return np.zeros(3), Rotation.from_euler("xyz", [0, 0, 0]).as_matrix()

    def coordinates_of_point_in_root_frame(self, point: np.ndarray) -> np.ndarray:
        return point

    @abc.abstractmethod
    def _build(self, *args, **kwargs) -> None:
        raise NotImplementedError


class MJCFSubComponent(ABC):
    def __init__(
        self,
        parent: Union[MJCFSubComponent, MJCFRootComponent],
        name: str,
        position: Optional[np.array] = None,
        euler: Optional[np.array] = None,
        *args,
        **kwargs,
    ) -> None:
        self._parent = parent
        self._name = name
        self._mjcf_body = parent.mjcf_body.add(
            "body", name=self._name, pos=position, euler=euler
        )
        self._coordinate_frame_in_root_frame = None
        self._build(*args, **kwargs)

    @property
    def mjcf_model(self) -> mjcf.RootElement:
        return self._parent.mjcf_model

    @property
    def mjcf_body(self) -> mjcf.Element:
        return self._mjcf_body

    @property
    def base_name(self) -> str:
        return self._name

    def attach(
        self,
        other: MJCFRootComponent,
        position: Optional[np.ndarray] = None,
        euler: Optional[np.ndarray] = None,
    ) -> None:
        attachment_site = self.mjcf_body.add(
            "site",
            name=f"{self.base_name}_attachment_{other.base_name}",
            pos=position,
            euler=euler,
        )
        other._parent = self
        frame = attachment_site.attach(other.mjcf_model)
        return frame

    @property
    def coordinate_frame_in_root_frame(self) -> (np.ndarray, np.ndarray):
        """
        Returns the MJCFSubComponent's coordinate frame with respect to the MJCFRootComponent's frame:
            0 -> the object's coordinates in root's frame
            1 -> a rotation matrix that represents the transformation from the object's local frame to the root's frame
        :return:
        """
        if self._coordinate_frame_in_root_frame is None:
            parent_origin, parent_rot = self._parent.coordinate_frame_in_root_frame

            pos = self.mjcf_body.pos
            euler = self.mjcf_body.euler
            my_rot = Rotation.from_euler("xyz", euler).as_matrix()

            self._coordinate_frame_in_root_frame = parent_origin + parent_rot.dot(
                pos
            ), parent_rot.dot(my_rot)
        return self._coordinate_frame_in_root_frame

    def coordinates_of_point_in_root_frame(self, point: np.ndarray) -> np.ndarray:
        """Returns the world coordinates of the given point in local coordinates"""
        origin, rot = self.coordinate_frame_in_root_frame
        return origin + rot.dot(point)

    @property
    def world_coordinates(self) -> np.ndarray:
        return self.coordinate_frame_in_root_frame[0]

    @abc.abstractmethod
    def _build(self, *args, **kwargs) -> None:
        raise NotImplementedError
