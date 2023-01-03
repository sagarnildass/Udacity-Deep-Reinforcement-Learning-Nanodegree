import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import random
from torchvision import transforms
from torch.autograd import Variable
from model_pixels import QNetwork
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 5e-4
BATCH_SIZE = 64
BUFFER_SIZE = int(1e5)
UPDATE_RATE = 4
GAMMA = 0.99
TAU = 1e-3


class Agent:

    def __init__(self,
                 action_size,
                 seed,
                 in_channels=4,
                 buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE,
                 device=device,
                 lr=LR,
                 gamma=GAMMA,
                 tau=TAU,
                 update_every=UPDATE_RATE,
                 update_target_every=1000,
                 use_double_q=False):
        """Agent.

        DQN/DDQN agent.
        """
        self.action_size = action_size
        self.seed = seed
        self.lr = lr
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.update_every = update_every
        self.update_target_every = update_target_every
        self.gamma = gamma
        self.tau = tau
        self.use_double_q = use_double_q

        # Q-Network
        self.qnetwork_local = QNetwork(action_size, seed, in_channels=in_channels).to(self.device)
        self.qnetwork_target = QNetwork(action_size, seed, in_channels=in_channels).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr, weight_decay=1e-4)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size, seed, device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        """step.

        (For training). This will add the current
        sars tuple in the buffer and it will perform a learning
        step over the QNetworks as per definition.
        """
        # save to experience buffer
        self.memory.add(state, action, reward, next_state, done)

        # update every n time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def learn(self, experiences):
        """learn.

        (For training). This will perform
        the actual learning steps over the QNetworks.
        """
        states, actions, next_states, rewards, dones = experiences

        # compute the targets as per formula
        Q_targets_next = self.qnetwork_target(next_states).detach()

        if self.use_double_q:
            Q_local_actions = self.qnetwork_local(next_states).detach().argmax(1)
            Q_targets_next = Q_targets_next.gather(1, Q_local_actions.unsqueeze(1))
            # Q_targets_next = Q_targets_next[Q_local_actions]
        else:
            Q_targets_next = Q_targets_next.max(1)[0].unsqueeze(1)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # compute expected Q values from the local Q network
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # backprop step
        self.optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 50)
        self.optimizer.step()

        # update the target network weights
        #if self.t_step & self.update_target_every == 0:
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local, target, tau):
        """soft update.

        The idea here is to update target weights slighlty in the direction
        of the local network. The tau parameter determines what proportion
        of each network to use. Small values of tau makes the target network
        weights more important than local.
        """
        for target_param, local_param in zip(target.parameters(), local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)

    def act(self, state, eps=0.):
        """act.

        Selects an action depending on the provided state
        by doing a forward pass over the internal Q network.
        """
        state = np.expand_dims(state, axis=0)
        #         state = state.squeeze(0)
        state = torch.from_numpy(state).float().to(self.device)
        self.qnetwork_local.eval()

        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.randint(self.action_size)


class ReplayBuffer:
    """Replay buffer.

    This class implements a simple replay
    buffer of a fixed size and able to
    take random samples of a fixed size.
    """

    def __init__(self, buffer_size, batch_size, seed, device):
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.device = device

        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Sample.

        Selects randomly a batch of sise batch_size
        from the current memory buffer. It delivers a
        tuple of (states, actions, next_states, dones)
        every element as a torch tensor.
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([np.expand_dims(e.state, axis=0) for e in experiences if e is not None])).float().to(self.device)
        #         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        n_states = torch.from_numpy(
            np.vstack([np.expand_dims(e.next_state, axis=0) for e in experiences if e is not None])).float().to(
            self.device)
        #         n_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, n_states, rewards, dones)

    def __len__(self):
        return len(self.memory)