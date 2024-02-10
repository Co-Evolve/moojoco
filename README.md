<h1>
  <a href="#"><img alt="moojoco-banner" src="https://github.com/Co-Evolve/moojoco-env/blob/main/assets/banner.png?raw=true" width="100%"/></a>
</h1>

**MooJoCo** provides a unified framework for implementing and interfacing with MuJoCo and MuJoCo-XLA simulation
environments.
The main goal of this framework is to **unify** the development and interfaces of environments implemented in native
MuJoCo (MJC) and MuJoCo-XLA (MJX).

## MJCFMorphology and MJCFArena

* Both are parameterized/reconfigurable MJCF (MuJoCo-XML) generators that serve as input to an environment.
* MJCFMorphology defines the robot morphology.
    * MJCFMorphologies are parameterized using
      the [Framework for Parameterized Robot Specifications (FPRS)](https://github.com/Co-Evolve/fprs).
    * MJCFMorphologies follow a modular design, dividing the robot into distinct parts.
* MJCFArena defines the arena in which the robot is places (i.e. all non-morphological structures).
    * MJCFArena are reconfigurable via a `ArenaConfiguration`
        * reconfigurable via a configuration
          [requirements.txt](requirements.txt)

## Unified MJC and MJX environment interface

* Reconfigurable through an environment configuration
* Functional programming
    * MJX requires the functional programming paradigm. To unify MJC and MJX, we thus apply this stronger coding
      constraint to the MJC side as well.
    * Simply said: functions are pure, and should not have side effects.
    * Environments expect states and return states.
* Gymnasium-like interface
    * Environments provide an observation and action space (automatically)
    * Outside code interacts with the environment using the typical `step`, `reset`, `render` and `close` functions.
    * Main difference with the original gymnasium interface: here the environment returns and expects states, which
      encapsulates the original observations, rewards, info, etc. in a datastructure, together with the underlying
      physics state.
* Differences between MJCEnv (i.e. a native MuJoCo environment) and MJXEnv (i.e. an MJX environment):
    * development: MJXEnv's should be implemented in JAX, MJCEnv's do not necessarily require JAX.
    * usage: MJXEnvState contains JAX arrays (and the additional mjx.Model and mjx.Data structures), the MJCEnvState
      uses numpy arrays.
* Observations are implemented using Observables; these define retriever functions that extract the observation based on
  the state datastructure and provide an easy way to implement observations.
* DualMuJocoEnvironment provides the unification of MJC and MJX, and allows conditional environment creation based on
  a backend string.
* Both MJC and MJX support human-mode and rgb_array rendering modes.
    * Note: MJX rendering is slow due to the offloading of datastructures of the GPU

## Examples

For practical applications and demonstrations of **MooJoCo**, please refer to
the [Bio-inspired Robotics Testbed](https://github.com/Co-Evolve/brt),
which employs this framework extensively.

## Installation

``pip install moojoco``