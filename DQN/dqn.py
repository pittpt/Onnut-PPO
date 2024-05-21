from torch import nn
import torch
import os
import msgpack


class Network(nn.Module):
    def __init__(self, gamma,input_dim, actions):
        super().__init__()
        self.gamma = gamma
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = nn.Sequential(
                                nn.Linear(input_dim, 64),
                                nn.Linear(64, actions))
        # print(self.net.summary())
        named_layers = dict(self.net.named_modules())
        # print(named_layers)

    def forward(self, x):
        return self.net(x)

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()
        return action

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(save_path,'wb') as f:
            f.write(params_data)

    def load(self,load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)
        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())
        params = {k: torch.as_tensor(v, device=self.device) for k,v in params_numpy.items()}

        self.load_state_dict(params)