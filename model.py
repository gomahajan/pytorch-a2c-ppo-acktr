import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical, DiagGaussian
from utils import init, init_normc_

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, recurrent_policy, num_actors):
        super(Policy, self).__init__()
        if len(obs_shape) == 3:
            self.base = CNNBase(obs_shape[0], recurrent_policy)
        elif len(obs_shape) == 1:
            assert not recurrent_policy, \
                "Recurrent policy is not implemented for the MLP controller"
            self.base = MLPBase(obs_shape[0])
        else:
            raise NotImplementedError

        #START: HACK
        self.base = FactoredMLPBase(obs_shape[0], num_actors)
        self.ddist = Categorical(self.base.output_size, self.base.num_actors)
        self.ddist2 = Categorical(self.base.output_size, self.base.num_actors)
        #END: HACK

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.state_size = self.base.state_size

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, states, hidden_decision, hidden_decision2 = self.base(inputs, states, masks)
        ddist = self.ddist(hidden_decision)
        choice = ddist.sample()
        choice_log_prob = ddist.log_probs(choice)

        ddist2 = self.ddist2(hidden_decision2)
        choice2 = ddist2.sample()
        choice_log_prob2 = ddist2.log_probs(choice2)

        hidden_actor = torch.empty(choice.shape[0], self.base.output_size)

        for i in range(0, inputs.shape[0]):
            hidden_actor[i] = self.base.actors[choice[i]](inputs[i]) + self.base.actors2[choice2[i]](inputs[i])

        dist = self.dist(hidden_actor)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean() + ddist.entropy().mean()

        return value, action, (choice, choice2), action_log_probs, (choice_log_prob, choice_log_prob2), states

    def get_value(self, inputs, states, masks):
        value, _, _, _ = self.base(inputs, states, masks)
        return value

    def evaluate_actions(self, inputs, states, masks, action, choicetuple):
        value, states, hidden_decision, hidden_decision2 = self.base(inputs, states, masks)
        ddist = self.ddist(hidden_decision)
        ddist2 = self.ddist2(hidden_decision2)
        choice, choice2 = choicetuple
        hidden_actor = torch.empty(inputs.shape[0], self.base.output_size)

        for i in range(0, inputs.shape[0]):
            hidden_actor[i] = self.base.actors[choice[i]](inputs[i]) + self.base.actors2[choice2[i]](inputs[i])

        dist = self.dist(hidden_actor)
        action_log_probs = dist.log_probs(action)

        choice_log_probs = ddist.log_probs(choice)

        dist_entropy = dist.entropy().mean() + ddist.entropy().mean()
        choice_log_probs2 = ddist2.log_probs(choice2)

        return value, action_log_probs, (choice_log_probs, choice_log_probs2), dist_entropy, states


class CNNBase(nn.Module):
    def __init__(self, num_inputs, use_gru):
        super(CNNBase, self).__init__()

        init_ = lambda m: init(m,
                      nn.init.orthogonal_,
                      lambda x: nn.init.constant_(x, 0),
                      nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, 512)),
            nn.ReLU()
        )

        if use_gru:
            self.gru = nn.GRUCell(512, 512)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

        init_ = lambda m: init(m,
          nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        self.train()

    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return 512
        else:
            return 1

    @property
    def output_size(self):
        return 512

    def forward(self, inputs, states, masks):
        x = self.main(inputs / 255.0)

        if hasattr(self, 'gru'):
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)

        return self.critic_linear(x), x, states


class MLPBase(nn.Module):
    def __init__(self, num_inputs):
        super(MLPBase, self).__init__()

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(64, 1))

        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor, states, torch.Tensor([0])

class FactoredMLPBase(nn.Module):
    def __init__(self, num_inputs, num_actors):
        super(FactoredMLPBase, self).__init__()
        self.num_actors = num_actors

        init_ = lambda m: init(m,
              init_normc_,
              lambda x: nn.init.constant_(x, 0))

        self.decider = nn.Sequential(
                init_(nn.Linear(num_inputs, 64)),
                nn.Tanh(),
                init_(nn.Linear(64, 64)),
                nn.Tanh()
        )

        self.decider2 = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.actors = []

        for i in range(0,self.num_actors):
            self.actors.append(nn.Sequential(
                init_(nn.Linear(num_inputs, 64))
            ))

        self.actors2 = []

        for i in range(0, self.num_actors):
            self.actors2.append(nn.Sequential(
                init_(nn.Linear(num_inputs, 64))
            ))

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, 64)),
            nn.Tanh(),
            init_(nn.Linear(64, 64)),
            nn.Tanh()
        )

        self.critic_linear = init_(nn.Linear(64, 1))

        self.train()

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 64

    def forward(self, inputs, states, masks):
        hidden_critic = self.critic(inputs)
        hidden_decision = self.decider(inputs)
        hidden_decision2 = self.decider2(inputs)

        return self.critic_linear(hidden_critic), states, hidden_decision, hidden_decision2