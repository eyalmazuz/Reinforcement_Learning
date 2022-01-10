import tensorflow as tf
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
