# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for meta-cfr algorithm."""

import functools
from typing import List
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from open_spiel.python.examples.meta_cfr.sequential_games.typing import ApplyFn
from open_spiel.python.examples.meta_cfr.sequential_games.typing import InfostateMapping
from open_spiel.python.examples.meta_cfr.sequential_games.typing import InfostateNode
from open_spiel.python.examples.meta_cfr.sequential_games.typing import Params


def get_batched_input(input_list: List[jax.numpy.DeviceArray],
                      infostate_list: List[InfostateNode],
                      illegal_action_list: List[List[int]], batch_size: int):
  """Returns list of function arguments extended to be consistent with batch size.

  Args:
    input_list: List of DeviceArrays.
    infostate_list: List of information state nodes.
    illegal_action_list: List of List of illegal actions. Each internal list
      contains illegal actions in each information state.
    batch_size: Batch size.

  Returns:
    input_list, infostate_list, and illegal_action_list with a size consistent
    with batch size (the size of returned arrays are multipliers of batch size).
  """
  items_to_sample = batch_size * (int(len(input_list) / batch_size) +
                                  1) - len(input_list)
  idx_sample = np.random.choice(len(input_list), items_to_sample)
  input_zip = np.array(
      list(zip(input_list, infostate_list, illegal_action_list)))
  input_lst_sample = input_zip[idx_sample]
  input_sample, infostate_sample, illegal_action_sample = zip(*input_lst_sample)

  input_list.extend(list(input_sample))
  infostate_list.extend(list(infostate_sample))
  illegal_action_list.extend(list(illegal_action_sample))
  return input_list, infostate_list, illegal_action_list


def mask(cfvalues: np.ndarray, infoset: List[InfostateNode], num_actions: int,
         batch_size: int) -> np.ndarray:
  """Returns counterfactual values of legal actions and put 0 for illegal ones.

  Args:
    cfvalues: Numpy array of counterfactual values.
    infoset: List of information states.
    num_actions: Number of possible actions to take.
    batch_size: Batch size.

  Returns:
    Masked counterfactual values. The counterfactual values of legal actions are
    kept as passed to this function and for illegal actions, we consider 0
    counterfactual value.
  """
  legal_actions = [[infoset[i].world_state.state.legal_actions()] *
                   cfvalues.shape[1] for i in range(batch_size)]

  masked_cfvalues = np.zeros(shape=[batch_size, cfvalues.shape[1], num_actions])
  for i in range(cfvalues.shape[0]):
    for j in range(cfvalues.shape[1]):
      np.put(masked_cfvalues[i][j], legal_actions[i][j], cfvalues[i][j])

  return np.stack(masked_cfvalues)


def filter_terminal_infostates(infostates_map: InfostateMapping):
  """Filter out terminal infostate_node values."""
  return {
      infostate_string: infostate_node
      for infostate_string, infostate_node in infostates_map.items()
      if not infostate_node.is_terminal()
  }


def get_network_output(net_apply: ApplyFn, net_params: Params,
                       net_input: np.ndarray, illegal_actions: List[int],
                       key: hk.PRNGSequence) -> jax.numpy.DeviceArray:
  """Returns policy generated as output of model.

  Args:
    net_apply: Haiku apply function.
    net_params: Haiku network parameters.
    net_input: Input of the model.
    illegal_actions: List of illegal actions we use to mask the model output.
    key: Pseudo random number.

  Returns:
    Policy generated by model. Model output is filtered to mask illegal actions.
  """
  net_output = jax.jit(net_apply)(net_params, key, net_input)

  if illegal_actions:
    net_output = jnp.delete(net_output, np.array(illegal_actions))

  return jax.nn.softmax(net_output)


def get_network_output_batched(
    net_apply: ApplyFn, net_params: Params, net_input: np.ndarray,
    all_illegal_actions: List[List[int]],
    key: hk.PRNGSequence) -> List[jax.numpy.DeviceArray]:
  """Returns policy of batched input generated as output of model.

  Args:
    net_apply: Haiku apply function.
    net_params: Haiku network parameters.
    net_input: Input of the model.
    all_illegal_actions: Nested list of illegal actions we use to mask the model
      output. Length of outer list is equal to the batch size.
    key: Pseudo random number.

  Returns:
    List of policies generated by model. Model output is filtered to mask
    illegal actions. Length of the returned list is equal to batch size.
  """
  net_output_batched = net_apply(net_params, next(key), net_input)

  batch_policies = []
  for i, illegal_actions in enumerate(all_illegal_actions):
    net_output = net_output_batched[i]
    if illegal_actions:
      net_output = jnp.expand_dims(
          jnp.delete(net_output, jnp.array(illegal_actions)), axis=0)

    batch_policies.append(jax.nn.softmax(net_output))
  return batch_policies


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 7, 9))
def meta_loss(net_params: Params, cfvalues: np.ndarray,
              net_apply: ApplyFn, steps: int, num_all_actions: int,
              infosets: List[InfostateNode],
              infostate_map: InfostateMapping,
              batch_size: int,
              key: hk.PRNGSequence,
              use_infostate_representation: bool = True) -> float:
  """Meta learning loss function.

  Args:
    net_params: Network parameters.
    cfvalues: Counterfactual values.
    net_apply: Haiku apply function.
    steps: Number of unrolling steps.
    num_all_actions: Number of actions.
    infosets: List of information states.
    infostate_map: Mapping from information state string to information state
      node.
    batch_size: Batch size.
    key: Pseudo random number.
    use_infostate_representation: Boolean value indicating if information state
      representation is used as part of input.

  Returns:
    Mean meta learning loss value.
  """
  regret_sum = np.zeros(shape=[batch_size, 1, num_all_actions])
  total_loss = 0
  step = 0
  infostate_str_one_hot = jnp.expand_dims(
      jnp.array([
          jax.nn.one_hot(infostate_map[infoset.infostate_string],
                         len(infostate_map)) for infoset in infosets
      ]),
      axis=1)

  def scan_body(carry, x):
    del x  # Unused
    regret_sum, current_step, total_loss = carry
    average_regret = regret_sum / (current_step + 1)

    if use_infostate_representation:
      net_input = jnp.concatenate((average_regret, infostate_str_one_hot),
                                  axis=-1)
    else:
      net_input = average_regret
    next_step_x = jax.jit(net_apply)(net_params, key, net_input)
    strategy = jax.nn.softmax(next_step_x)

    value = jnp.matmul(
        jnp.array(cfvalues), jnp.transpose(strategy, axes=[0, 2, 1]))
    curren_regret = jnp.array(cfvalues) - value
    regret_sum += jnp.expand_dims(jnp.mean(curren_regret, axis=1), axis=1)
    current_loss = jnp.mean(
        jnp.max(
            jax.numpy.concatenate(
                [regret_sum,
                 jnp.zeros(shape=[batch_size, 1, 1])],
                axis=-1),
            axis=-1))
    total_loss += current_loss
    current_step += 1
    return (regret_sum, current_step, total_loss), None

  (regret_sum, step, total_loss), _ = jax.lax.scan(
      scan_body, (regret_sum, step, total_loss), None, length=steps)
  return total_loss
