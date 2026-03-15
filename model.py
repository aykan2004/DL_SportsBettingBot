# model.py
import torch
import torch.nn as nn

class SuperSoccerNet(nn.Module):
    def __init__(self, num_teams, num_leagues):
        super(SuperSoccerNet, self).__init__()
        self.team_emb = nn.Embedding(num_teams, 16)
        self.league_emb = nn.Embedding(num_leagues, 4)
        self.cont_layer = nn.Linear(6, 16)
        self.fc1 = nn.Linear(52, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x_cat, x_cont):
        h = self.team_emb(x_cat[:, 0])
        a = self.team_emb(x_cat[:, 1])
        l = self.league_emb(x_cat[:, 2])
        c = self.relu(self.cont_layer(x_cont))
        combined = torch.cat([h, a, l, c], dim=1)
        x = self.dropout(self.relu(self.bn1(self.fc1(combined))))
        x = self.relu(self.fc2(x))
        return self.fc3(x)