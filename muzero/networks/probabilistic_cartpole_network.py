import math

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import regularizers, Sequential
from tensorflow.keras.layers import Dense

from game.game import Action
from networks.network import BaseNetwork

tfd = tfp.distributions
tfpl = tfp.layers

negative_log_likelihood = lambda y, rv_y: -rv_y.log_prob(y)

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  c = np.log(np.expm1(1.))
  return Sequential([
      tfp.layers.VariableLayer(2 * n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t[..., :n],
                     scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
          reinterpreted_batch_ndims=1)),
  ])

def prior_trainable(kernel_size, bias_size=0, dtype=None):
  n = kernel_size + bias_size
  return Sequential([
      tfp.layers.VariableLayer(n, dtype=dtype),
      tfp.layers.DistributionLambda(lambda t: tfd.Independent(
          tfd.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
  ])

class ProbabilisticCartPoleNetwork(BaseNetwork):

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
        # dynamic_network = Sequential([
        #   tfp.layers.DenseVariational(hidden_neurons, posterior_mean_field, prior_trainable, kl_weight=0.001,
        #                     activation='relu', activity_regularizer=regularizer),
        #   tfp.layers.DenseVariational(representation_size, posterior_mean_field, prior_trainable, kl_weight=0.001,
        #                     activation=representation_activation, activity_regularizer=regularizer),
        # ])

        # dynamic_network = Sequential([Dense(hidden_neurons, activation='relu', kernel_regularizer=regularizer),
        #                               Dense(representation_size, activation=representation_activation,
        #                                     kernel_regularizer=regularizer)])
        dynamic_network = Sequential([
            tfpl.DenseVariational(hidden_neurons, posterior_mean_field, prior_trainable, activation='relu', activity_regularizer=regularizer),
            tfpl.DenseVariational(representation_size + representation_size, posterior_mean_field, prior_trainable, activation=representation_activation,
                                            activity_regularizer=regularizer),
            tfpl.DistributionLambda(lambda t: tfd.Independent(
                tfd.Normal(loc=t[..., :representation_size],
                            scale=1e-8 + tf.nn.softplus(np.log(np.expm1(1.)) + t[..., representation_size:])),
                reinterpreted_batch_ndims=1)),
        ])
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
