"""
Taken from https://github.com/RobertTLange/gymnax, with some slight modifications:
- gym -> gymnasium
- types:
    - Accept array types for Box.low and Box.high as well
- Added __repr__
"""

from collections import OrderedDict
from typing import Any, Dict, Sequence, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces as gspc


def _short_repr(arr: jnp.ndarray) -> str:
    if arr.size != 0 and jnp.min(arr) == jnp.max(arr):
        return str(jnp.min(arr))
    return str(arr)


class Space:
    """
    Minimal jittable class for abstract space.
    """

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        raise NotImplementedError

    def contains(self, x: jnp.int_) -> bool:
        raise NotImplementedError


class Discrete(Space):
    """
    Minimal jittable class for discrete space.
    """

    def __init__(self, num_categories: int):
        assert num_categories >= 0
        self.n = num_categories
        self.shape = ()
        self.dtype = jnp.int_

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=0, maxval=self.n
        ).astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class MultiDiscrete(Space):
    """
    Minimal jittable class for MultiDiscrete space.
    Assumes integer categories starting from 0
    """

    def __init__(self, num_categories_per_dim: Sequence[int]) -> None:
        self.num_categories_per_dim = jnp.array(num_categories_per_dim).astype(int)
        self.shape = self.num_categories_per_dim.shape

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng,
            shape=self.shape,
            minval=jnp.zeros(self.shape),
            maxval=self.num_categories_per_dim,
        )

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.

        Returns:
            A representation of the space
        """
        return f"MultiDiscrete({(_short_repr(self.num_categories_per_dim))}, {self.shape}, {self.dtype})"


class Box(Space):
    """
    Minimal jittable class for array-shaped spaces.
    TODO: Add unboundedness - sampling from other distributions, etc.
    """

    def __init__(
        self,
        low: float | jax.Array,
        high: float | jax.Array,
        shape: Tuple[int],
        dtype: jnp.dtype = jnp.float32,
    ):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from 1D continuous range."""
        return jax.random.uniform(
            rng, shape=self.shape, minval=self.low, maxval=self.high
        ).astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(jnp.all(x >= self.low), jnp.all(x <= self.high))
        return range_cond

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        return f"Box({_short_repr(self.low)}, {_short_repr(self.high)}, {self.shape}, {self.dtype})"


class Dict(Space):
    """Minimal jittable class for dictionary of simpler jittable spaces."""

    def __init__(self, spaces: Dict[Any, Space]):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, rng: chex.PRNGKey) -> Dict:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return OrderedDict(
            [
                (k, self.spaces[k].sample(key_split[i]))
                for i, k in enumerate(self.spaces)
            ]
        )

    def contains(self, x: jnp.int_) -> bool:
        """Check whether dimensions of object are within subspace."""
        # type_cond = isinstance(x, Dict)
        # num_space_cond = len(x) != len(self.spaces)
        # Check for each space individually
        out_of_space = 0
        for k, space in self.spaces.items():
            out_of_space += 1 - space.contains(getattr(x, k))
        return out_of_space == 0

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return (
            "Dict(" + ", ".join([f"{k!r}: {s}" for k, s in self.spaces.items()]) + ")"
        )


class Tuple(Space):
    """Minimal jittable class for tuple (product) of jittable spaces."""

    def __init__(self, spaces: Sequence[Space]):
        self.spaces = spaces
        self.num_spaces = len(spaces)

    def sample(self, rng: chex.PRNGKey) -> Tuple[chex.Array]:
        """Sample random action from all subspaces."""
        key_split = jax.random.split(rng, self.num_spaces)
        return tuple([s.sample(key_split[i]) for i, s in enumerate(self.spaces)])

    def contains(self, x: jnp.int_) -> bool:
        """Check whether dimensions of object are within subspace."""
        # type_cond = isinstance(x, tuple)
        # num_space_cond = len(x) != len(self.spaces)
        # Check for each space individually
        out_of_space = 0
        for i, space in enumerate(self.spaces):
            out_of_space += 1 - space.contains(x[i])
        return out_of_space == 0

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"
