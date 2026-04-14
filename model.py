# model.py
import torch
import torch.nn as nn


class SuperSoccerNet(nn.Module):
    def __init__(self, num_teams, num_leagues):
        super(SuperSoccerNet, self).__init__()

        # --- STATIC EMBEDDINGS ---
        self.team_embed = nn.Embedding(num_teams, 16)
        self.league_embed = nn.Embedding(num_leagues, 4)

        # --- RECURRENT BRANCH (MOMENTUM) ---
        # Input size = 4 (e.g., Home GF, Home GA, Away GF, Away GA)
        # Hidden size = 16 (Extracting the final state h_t)
        self.lstm = nn.LSTM(input_size=4, hidden_size=16, batch_first=True)

        # --- MERGER & CLASSIFICATION ---
        # Home(16) + Away(16) + League(4) + LSTM(16) + Cont(6) = 58 dimensions
        self.fc_layers = nn.Sequential(
            nn.Linear(58, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 3)
        )

    def forward(self, h_idx, a_idx, l_idx, seq_data, cont_data):
        # 1. Pass categorical data through embeddings
        h_emb = self.team_embed(h_idx)
        a_emb = self.team_embed(a_idx)
        l_emb = self.league_embed(l_idx)

        # 2. Pass the sequence data through the LSTM
        # seq_data shape: (Batch, 5 matches, 4 features)
        lstm_out, (h_n, c_n) = self.lstm(seq_data)

        # Extract the final hidden state of the sequence
        lstm_h = h_n.squeeze(0)

        # 3. Concatenate everything into the 58-dim tensor
        combined = torch.cat([h_emb, a_emb, l_emb, lstm_h, cont_data], dim=1)

        # 4. Pass through the dense feedforward network
        output = self.fc_layers(combined)

        return output