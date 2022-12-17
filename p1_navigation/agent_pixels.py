import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torch.autograd import Variable
from model import QPixelNetwork
from collections import deque, namedtuple
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR = 5e-3
BATCH_SIZE = 215
BUFFER_SIZE = int(1e5)
UPDATE_RATE = 4
GAMMA = 0.99
TAU = 1e-3


class Agent():
    def __init__(self, state_size, action_size, seed, training, lr=LR):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0.

        print('loaded cnn network')
        self.loader = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.QN_local = QPixelNetwork(state_size, action_size, seed, training).to(device)
        self.QN_target = QPixelNetwork(state_size, action_size, seed, training).to(device)
        self.optimizer = optim.Adam(self.QN_local.parameters(), lr=lr)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, device)  # TODO

    def act(self, state, eps):
        #         if self.pixels is True:

        # state = Variable(torch.from_numpy(state).float().to(device).view(state.shape[0],3,32,32))

        self.QN_local.eval()

        if torch.no_grad():
            action_values = self.QN_local(state)
        self.QN_local.train()
        if random.random() > eps:
            return int(np.argmax(action_values.cpu().data.numpy()))
        else:
            return int(random.choice(np.arange(self.action_size)))

    def step(self, state, action, reward, next_state, done, stack_size):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % UPDATE_RATE
        if self.t_step == 0 and len(self.memory) > BATCH_SIZE:
            samples = self.memory.sample()
            self.learn(samples, GAMMA, stack_size)

    def learn(self, experiences, gamma, stack_size):
        states, actions, rewards, next_states, dones = experiences

        if self.pixels:
            next_states = Variable(
                next_states)  # next_states.view(next_states.shape[0],stack_size,3, stack_size,32,32))
            states = Variable(states)  # states.view(states.shape[0],3,64,64))
        #         else:
        # todo bring back the old version stuff here

        _target = self.QN_target(next_states).detach().max(1)[0].unsqueeze(1)
        action_values_target = rewards + gamma * _target * (1 - dones)
        action_values_expected = self.QN_local(states).gather(1, actions)

        loss = F.mse_loss(action_values_expected, action_values_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target Qnetwork
        for target_param, local_param in zip(self.QN_target.parameters(), self.QN_local.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        self.seed = random.seed(seed)
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=[
            "state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)