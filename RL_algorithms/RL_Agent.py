from abc import ABC, abstractmethod
import torch


class Abstract_agent(ABC):
    """
    Abstract class for trained agents
    """

    def __init__(self, model, scaler):
        self.model = model
        self.model.eval()
        self.scaler = scaler

    @abstractmethod
    def get_action(self, state):
        pass


class DQN_agent(Abstract_agent):
    def __init__(self, model, scaler):
        super().__init__(model, scaler)

    def get_action(self, state):
        if self.scaler is not None:
            state = self.scaler.transform([state])
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.model(state).max(1).indices.view(1, 1)
        return action.item()
