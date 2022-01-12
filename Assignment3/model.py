import tensorflow as tf
from tensorflow.core.protobuf.config_pb2 import _CONFIGPROTO_DEVICECOUNTENTRY
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class MLP(Model):
    def __init__(self, input_dim , output_dim, hidden_dim, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hiddens = [Dense(hidden_dim, activation='relu') for _ in range(n_layers)]

        self.logits = Dense(output_dim, activation='linear') 

    def call(self, x):
        for layer in self.hiddens:
            x = layer(x)
        output = self.logits(x)
        return output 

class ActorCritic():

    def __init__(self, input_dim , output_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.actor = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=32, n_layers=1)
        self.critic = MLP(input_dim=input_dim, output_dim=1, hidden_dim=12, n_layers=1)

    def __call__(self, state):

        output_value = self.critic(state)

        action_probs = self.actor(state)

        return output_value, action_probs

class ProgressiveActorCritic():

    def __init__(self, actors, critics, input_dim , output_dim, hidden_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.actors = actors
        self.critics = critics

        self.actor_adapter = Dense(output_dim, activation='linear')
        self.critic_adapter = Dense(1, activation='linear')

        self.actor = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=32, n_layers=1)
        self.critic = MLP(input_dim=input_dim, output_dim=1, hidden_dim=12, n_layers=1)

    def __call__(self, state):

        output_value = self.critic(state)
        critic_results = self.get_prog(state, self.critics, self.critic_adapter)

        action_probs = self.actor(state)
        actor_results = self.get_prog(state, self.actors, self.actor_adapter)

        action_probs = action_probs + actor_results
        output_value = output_value + critic_results

        return output_value, action_probs


    def get_prog(self, state, models, adapter):
        results = None
        for model in models:
            x = state
            for layer in model.hiddens:
                x = layer(x)
            x = adapter(x)
            if results is None:
                results = x
            results = tf.add(results, x)

        return results

