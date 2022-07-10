import math

import numpy as np
from tensorflow.keras import regularizers, Sequential
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from tensorflow.keras.models import Model

from game.game import Action
from networks.network import BaseNetwork


class EnsembleCartPoleNetwork(BaseNetwork):

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 representation_size: int,
                 max_value: int,
                 hidden_neurons: int = 64,
                 weight_decay: float = 1e-4,
                 representation_activation: str = 'tanh'):
        self.state_size = state_size
        self.action_size = action_size
        self.value_support_size = math.ceil(math.sqrt(max_value)) + 1

        regularizer = regularizers.l2(weight_decay)
        representation_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                             Dense(representation_size, activation=representation_activation,
                                                   kernel_regularizer=regularizer)])
        value_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                    Dense(self.value_support_size, kernel_regularizer=regularizer)])
        policy_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
                                     Dense(action_size, kernel_regularizer=regularizer)])
        dynamic_network = EnsembleDynamicsModel(
          num_models=5,
          hidden_heurons=hidden_neurons,
          representation_size=representation_size,
          regularizer=regularizer)
        reward_network = Sequential([Dense(16, activation='relu', kernel_regularizer=regularizer),
                                     Dense(1, kernel_regularizer=regularizer)])

        super().__init__(representation_network, value_network, policy_network, dynamic_network, reward_network)

    def _value_transform(self, value_support: np.array) -> float:
        """
        The value is obtained by first computing the expected value from the discrete support.
        Second, the inverse transform is then apply (the square function).
        """

        value = self._softmax(value_support)
        value = np.dot(value, range(self.value_support_size))
        value = np.asscalar(value) ** 2
        return value

    def _reward_transform(self, reward: np.array) -> float:
        return np.asscalar(reward)

    def _conditioned_hidden_state(self, hidden_state: np.array, action: Action) -> np.array:
        conditioned_hidden = np.concatenate((hidden_state, np.eye(self.action_size)[action.index]))
        return np.expand_dims(conditioned_hidden, axis=0)

    def _softmax(self, values):
        """Compute softmax using numerical stability tricks."""
        values_exp = np.exp(values - np.max(values))
        return values_exp / np.sum(values_exp)




class EnsembleDynamicsModel(Model):
  def __init__(self, num_models, hidden_heurons, representation_size, regularizer) -> None:
      super(EnsembleDynamicsModel, self).__init__()
      self.num_models = num_models
      self.hidden_neurons = hidden_heurons
      self.models = []
      for _ in range(self.num_models):
        network = Sequential([
          Dense(self.hidden_neurons, activation='relu', kernel_regularizer=regularizer),
          Dense(representation_size, activation='relu', kernel_regularizer=regularizer)
        ])
        self.models.append(network)


  def call(self, input):
    outputs = []
    for model in self.models:
      output = model(input)
      outputs.append(output)

    outputs = np.array(outputs)
    return np.mean(outputs, axis=0)
    # variance = np.cov(outputs)



