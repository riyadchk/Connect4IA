# dqnagent.py
from .player import Player  # Make sure to adjust this import path as needed.
from .dqn import DQN
from .replaybuffer import ReplayBuffer
from game.board import Board
import torch
import numpy as np
import random


class DQNPlayer(Player):
    def __init__(
        self,
        piece,
        input_dim,
        output_dim=7,
        model_path=None,
        epsilon=1.0,
        gamma=0.95,
        lr=0.1,
    ):
        super().__init__(piece)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN()
        self.model.to(device)
        self.target_model = DQN()  # For more stable Q-targets
        self.target_model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.memory = ReplayBuffer(10000)  # Experience replay buffer
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.gamma = gamma
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.model.fc3.out_features)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values, dim=1).item()

    def replay(self, batch_size):
        self.model.train()
        if len(self.memory) < batch_size:
            return
        batch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state))
            target_f = self.model(state)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.model.eval()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def move(self, board):
        state = self._get_state_from_board(board)
        action = self.act(state)
        return action

    def _get_state_from_board(self, board):
        # Implement the conversion from board to state
        """Convert the board state to a 3D tensor suitable for DQN input."""
        state = np.zeros((6, 7, 2))
        if isinstance(board, Board):
            board = board.board
        for r in range(6):
            for c in range(7):
                if board[r][c] == self.piece:
                    state[r][c][0] = 1
                elif board[r][c] != 0:
                    state[r][c][1] = 1
        return state.flatten()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
