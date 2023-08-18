.. role:: python(code)
    :language: python

|

You have an environment, a PyTorch model, and a reinforcement learning library that are designed to work together but don't. PufferLib provides one-line wrappers that make them play nice.

.. card::
  :link: https://colab.research.google.com/drive/1l1qLjerLwYoLjuKNr9iVc3TZ8gW2QVnz?usp=sharing
  :width: 75%
  :margin: 4 2 auto auto
  :text-align: center

  **Click to Demo PufferLib in Colab**

|
.. raw:: html

    <center>
      <video width=100% height="auto" nocontrols autoplay playsinline muted loop>
        <source src="../_static/banner.webm" type="video/webm">
        <source src="../_static/banner.mp4" type="video/mp4">
        Your browser does not support this video.
      </video>
    </center>

.. raw:: html

    <div style="text-align: center;">
        <div style="display: flex; align-items: center; justify-content: center; margin: auto;">
            <div style="flex-shrink: 0; width: 60px; margin-right: 20px;">
                <a href="https://github.com/pufferai/pufferlib" target="_blank">
                    <img src="https://img.shields.io/github/stars/pufferai/pufferlib?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star pufferai/pufferlib" width="60px">
                </a>
            </div>
            <a href="https://discord.gg/puffer" target="_blank" style="margin-right: 20px;">
                <img src="https://dcbadge.vercel.app/api/server/puffer?style=plastic" alt="Discord">
            </a>
            <a href="https://twitter.com/jsuarez5341" target="_blank">
                <img src="https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40jsuarez5341" alt="Twitter">
            </a>
        </div>
    </div>

|

Join our community Discord for support and Discussion, follow my Twitter for news, and star the repo to feed the puffer.

.. dropdown:: Installation

  .. tab-set::
    
    .. tab-item:: PufferTank

      `PufferTank <https://github.com/pufferai/puffertank>`_ is a GPU container with PufferLib and dependencies for all environments in the registry, including some that are slow and tricky to install.

      If you are new to containers, clone the repository and open it in VSCode. You will need to install the Dev Container plugin as well as Docker Desktop. VSCode will then detect the settings in .devcontainer and set up the container for you.

    .. tab-item:: Pip

      PufferLib is also available as a standard pip package.

      .. code-block:: python
        
        pip install pufferlib

      To install additional environments and frameworks:

      .. code-block:: python
        
        pip install pufferlib[nmmo,cleanrl]

      Note that some environments require additional non-pip dependencies. Follow the additional setup from the maintainers of that environment, or just use PufferTank.
         
.. dropdown:: Contributors

   **Joseph Suarez**: Creator and developer of PufferLib

   **David Bloomin**: Policy pool/store/selector

   **Nick Jenkins**: Layout for the system architecture diagram. Adversary.design.

   **Andranik Tigranyan**: Streamline and animate the pufferfish. Hire him on UpWork if you like what you see here.

   **Sara Earle**: Original pufferfish model. Hire her on UpWork if you like what you see here.

User Guide
##########

**You can open this guide in a Colab notebook by clicking the demo button at the top of this page**

Complex environments may have heirarchical observations and actions, variable numbers of agents, and other quirks that make them difficult to work with and incompatible with standard reinforcement learning libraries. PufferLib's emulation layer makes every environment look like it has flat observations and actions and a constant number of agents, with no changes to the underlying environment. Here's how it works with two notoriously complex environments, NetHack and Neural MMO:

.. code-block:: python

    import pufferlib.emulation 
    import nle
    import nmmo

    nethack = pufferlib.emulation.GymPufferEnv(env_creator=nle.env.NLE)
    neural_mmo = pufferlib.emulation.PettingZooPufferEnv(env_creator=nle.env.NLE)

You can pass envs by class, creator function, or object, with or without additional arguments. These wrappers enable us to make some optimizations to vectorization code that would be difficult to implement otherwise. You can choose from a variety of vectorization backends. They all share the same interface with synchronous and asynchronous options:

.. code-block:: python

    import pufferlib.vectorization
 
    env_creator = lambda: pufferlib.emulation.GymPufferEnv(env_creator=nmmo.Env)

    envs = pufferlib.vectorization.Serial(
        env_creator, num_envs=2, num_workers=2)
    envs = pufferlib.vectorization.Multiprocessing(
        env_creator, num_envs=2, num_workers=2)
    envs = pufferlib.vectorization.Multiprocessing(
        env_creator, num_envs=2, num_workers=2)

    # Synchronous vectorization
    obs = envs.reset(seed=42)
    obs, rewards, dones, infos = envs.step(actions)

    # Asynchronous vectorization
    envs.async_reset()
    obs, rewards, dones, infos = envs.recv()
    envs.send(actions)

We suggest Serial for debugging and Multiprocessing for most training runs. Ray is a good option if you need to scale beyond a single machine. 

PufferLib allows you to write vanilla PyTorch policies and use them with multiple learning libraries. We take care of the details of converting between the different APIs. Here's a policy that will work with *any* environment, with a one-line wrapper for CleanRL:

.. code-block:: python

  from torch import nn

  import pufferlib.frameworks.cleanrl


  class Default(Policy):
      def __init__(self, envs, input_size=128, hidden_size=128):
          super().__init__()
          self.encoder = nn.Linear(np.prod(envs.single_observation_space.shape), hidden_size)
          self.decoders = nn.ModuleList([nn.Linear(hidden_size, n)
                  for n in envs.single_action_space.nvec])
          self.value_head = nn.Linear(hidden_size, 1)

      def forward(self, env_outputs):
          env_outputs = env_outputs.reshape(env_outputs.shape[0], -1)
          hidden = self.encoder(env_outputs)
          actions = [dec(hidden) for dec in self.decoders]
          value = self.value_head(hidden)
          return actions, value

  CleanRLPolicy = pufferlib.frameworks.cleanrl.CleanRL(Default(envs))

There's also a lightweight, fully optional base policy class for PufferLib. It breaks the forward pass into two functions, encode_observations and decode_actions. The advantage of this is that it lets us handle recurrance for you, since every framework does this a bit differently. Our actual default policy is implemented this way, so we'll use it as an example:

.. code-block:: python

  from torch import nn

  import pufferlib.frameworks.cleanrl
  import pufferlib.models

  policy = pufferlib.models.Default(envs)
  CleanRLPolicy = pufferlib.frameworks.cleanrl.CleanRL(policy)

So far, the code above is fully general and does not rely on PufferLib support for specific environments. For convenience, we also provide a registry of environments and models. Here's a complete example:

.. code-block:: python

  import torch
  import nle

  import pufferlib.emulation
  import pufferlib.vectorization
  import pufferlib.frameworks.cleanrl
  import pufferlib.models

  import pufferlib.registry.nethack


  make_env = pufferlib.registry.nethack.make_env
  envs = pufferlib.vectorization.Serial(
      env_creator=make_env, num_workers=2, envs_per_worker=2)

  policy = pufferlib.registry.nethack.Policy(envs)
  policy = pufferlib.models.Recurrent(policy)
  policy = pufferlib.frameworks.cleanrl.Policy(policy)

  # Standard environment loop
  obs = envs.reset(seed=42)
  for _ in range(32):
      actions = policy.get_action_and_value(torch.Tensor(obs))[0].numpy()
      obs, reward, done, info = envs.step(actions)

It's that simple -- almost. One small quirk is that, because PufferLib flattens observations, you have to unflatten them in the network forward pass:

.. code-block:: python

  env_outputs = pufferlib.emulation.unpack_batched_obs(
      env_outputs, self.envs.flat_observation_space
  )

Save a reference to your vectorized environments in the policy init method so you can access the flat observation space needed for this call.


Libraries
#########

PufferLib's emulation layer adheres to the Gym and PettingZoo APIs: you can use it with *any* environment and learning library (subject to Limitations). The libraries and environments below are just the ones we've tested. We also provide additional tools to make them easier to work with.

PufferLib provides *pufferlib.frameworks* for the the learning libraries below. These are short wrappers over your vanilla PyTorch policy that handles learning library API details for you. Additionally, if you use our *optional* model API, which just requires you to split your *forward* function into an *encode* and *decode* portion, we can handle recurrance for you. This is the approach we use in our default policies.

.. raw:: html

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/vwxyzjn/cleanrl" target="_blank">
                <img src="https://img.shields.io/github/stars/vwxyzjn/cleanrl?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star CleanRL" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/vwxyzjn/cleanrl">CleanRL</a> provides single-file RL implementations suited for 80+% of academic research. It was designed for simple environments like Atari, but with PufferLib, you can use it with just about anything.</p>
        </div>
    </div>


:ref:`Minimal CleanRL Demo:` Shows how to integrate PufferLib with minimal code changes. Most users should start here.

:ref:`CleanPuffeRL Demo:` Uses our heavily customized version of CleanRL PPO with optimizations for variable agent populations, self-play, and experiment management. Proudly supporting the NeurIPS 2023 Neural MMO Competition.


.. raw:: html

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/anyscale/ray" target="_blank">
                <img src="https://img.shields.io/github/stars/ray-project/ray?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Ray" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://docs.ray.io/">Ray</a> is a general purpose distributed computing framework that includes <a href="https://docs.ray.io/en/latest/rllib">RLlib</a>, an industry reinforcement learning library.</p>
        </div>
    </div>

:ref:`RLlib Demo:` Shows how to integrate PufferLib RLlib.

Environments
############

We also provide a registry of environments and models that are supported out of the box. These environments are already set up for you in PufferTank and are used in our test cases to ensure they work with PufferLib. Several also include reasonable baseline policies. Join our Discord if you would like to add setup and tests for new environments or improvements to any of the baselines.


.. raw:: html

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/openai/gym" target="_blank">
                <img src="https://img.shields.io/github/stars/openai/gym?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star OpenAI Gym" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/openai/gym">OpenAI Gym</a> is the standard API for single-agent reinforcement learning environments. It also contains some built-in environments. We include <a href="https://www.gymlibrary.dev/environments/box2d/">Box2D</a> in our registry.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Farama-Foundation/Arcade-Learning-Environment" target="_blank">
                <img src="https://img.shields.io/github/stars/Farama-Foundation/Arcade-Learning-Environment?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Arcade Learning Environment" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/Farama-Foundation/Arcade-Learning-Environment">Arcade Learning Environment</a> provides a Gym interface for classic Atari games. This is the most popular benchmark for reinforcement learning algorithms.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Farama-Foundation/PettingZoo" target="_blank">
                <img src="https://img.shields.io/github/stars/Farama-Foundation/PettingZoo?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star PettingZoo" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://pettingzoo.farama.org">PettingZoo</a> is the standard API for multi-agent reinforcement learning environments. It also contains some built-in environments. We include <a href="https://pettingzoo.farama.org/environments/butterfly/">Butterfly</a> in our registry.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/neuralmmo/environment" target="_blank">
                <img src="https://img.shields.io/github/stars/openai/neural-mmo?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Neural MMO" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://neuralmmo.github.io">Neural MMO</a> is a massively multiagent environment for reinforcement learning. It combines large agent populations with high per-agent complexity and is the most actively maintained (by me) project on this list.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/geek-ai/MAgent" target="_blank">
                <img src="https://img.shields.io/github/stars/geek-ai/MAgent?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star MAgent" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/geek-ai/MAgent/blob/master/doc/get_started.md">MAgent</a> is a platform for large-scale agent simulation.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/openai/procgen" target="_blank">
                <img src="https://img.shields.io/github/stars/openai/procgen?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Procgen" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/openai/procgen">Procgen</a> is a suite of arcade games for reinforcement learning with procedurally generated levels. It is one of the most computationally efficient environments on this list.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/facebookresearch/nle" target="_blank">
                <img src="https://img.shields.io/github/stars/facebookresearch/nle?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star NLE" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/facebookresearch/nle">Nethack Learning Environment</a> is a port of the classic game NetHack to the Gym API. It combines extreme complexity with high simulation efficiency.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/danijar/crafter" target="_blank">
                <img src="https://img.shields.io/github/stars/danijar/crafter?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Crafter" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/danijar/crafter">Crafter</a> is a top-down 2D Minecraft clone for RL research. It provides pixel observations and relatively long time horizons.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Bam4d/Griddly" target="_blank">
                <img src="https://img.shields.io/github/stars/Bam4d/Griddly?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star Griddly" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://griddly.readthedocs.io/en/latest/">Griddly</a> is an extremely optimized platform for building reinforcement learning environments. It also includes a large suite of built-in environments.</p>
        </div>
    </div>

    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <div style="flex-shrink: 0; width: 100px; margin-right: 20px;">
            <a href="https://github.com/Farama-Foundation/MicroRTS-Py" target="_blank">
                <img src="https://img.shields.io/github/stars/Farama-Foundation/MicroRTS-Py?labelColor=999999&color=66dcdc&cacheSeconds=100000" alt="Star MicroRTS-Py" width="100px">
            </a>
        </div>
        <div>
            <p><a href="https://github.com/Farama-Foundation/MicroRTS-Py">Gym MicroRTS</a> is a real time strategy engine for reinforcement learning research. The Java configuration is a bit finicky -- we're still debugging this.</p>
        </div>
    </div>

Current Limitations
###################

- No continuous action spaces (WIP)
- Pre-gymnasium Gym and PettingZoo only (WIP)
- Support for heterogenous observations and actions requires you to specify teams such that each team has the same observation and action space. There's no good way around this.

License
#######

PufferLib is free and open-source software under the MIT license. This is the full set of tools maintained by PufferAI; we do not have private repositories with additional utilities.

