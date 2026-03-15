import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

from model import SuperSoccerNet


class HistData(Dataset):
    def __init__(self, data_path, map_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        with open(map_path, 'r') as f:
            m = json.load(f)
            self.mu = np.array(m['scaler_mean'])
            self.sigma = np.array(m['scaler_scale'])
            self.l_map = {int(k): v for k, v in m['leagues'].items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]['features']

        # Guard against unmapped leagues
        l_idx = self.l_map.get(int(row['l_id']), 0)
        cats = torch.tensor([row['h_id'], row['a_id'], l_idx],
                            dtype=torch.long)

        stats = np.array(
            [[row['h_pts'], row['a_pts'], row['h_gd'], row['a_gd']]])
        norm = (stats - self.mu[:4]) / self.sigma[:4]
        imp = np.array([[row['h_imp'], row['a_imp']]])

        conts = torch.tensor(np.hstack((norm, imp))[0], dtype=torch.float32)
        y = torch.tensor(self.data[i]['label'], dtype=torch.long)
        odds = torch.tensor(self.data[i]['odds'], dtype=torch.float32)

        return cats, conts, y, odds


def run_sim(model, loader):
    print("\n--- Running Bankroll Sim ---")
    model.eval()

    br = 1000.0
    stake = 10.0
    bets = 0
    hits = 0

    with torch.no_grad():
        for cats, conts, y, odds in loader:
            out = torch.softmax(model(cats, conts), dim=1)

            for i in range(len(y)):
                probs = out[i].numpy()
                true_res = y[i].item()
                line = odds[i].numpy()

                pick = int(np.argmax(probs))
                conf = probs[pick]

                # Filter out low conviction bets
                if conf > 0.45:
                    bets += 1
                    if pick == true_res:
                        hits += 1
                        br += stake * (line[pick] - 1)
                    else:
                        br -= stake

    roi = ((br - 1000) / 1000) * 100
    win_rate = (hits / bets) * 100 if bets > 0 else 0

    print(f"Bets: {bets} | Win Rate: {win_rate:.1f}%")
    print(f"End Bankroll: ${br:.2f} | ROI: {roi:.2f}%\n")


def main():
    dataset = HistData('historical_dataset_2024.json', 'mappings.json')
    n = len(dataset)

    # Chronological split: past predicts future.
    # Don't use random_split to avoid data leakage.
    train_n = int(0.6 * n)
    val_n = int(0.2 * n)

    train_data = Subset(dataset, range(0, train_n))
    val_data = Subset(dataset, range(train_n, train_n + val_n))
    test_data = Subset(dataset, range(train_n + val_n, n))

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    test_loader = DataLoader(test_data, batch_size=64)

    with open('mappings.json', 'r') as f:
        m = json.load(f)

    net = SuperSoccerNet(len(m['teams']), len(m['leagues']))
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

    train_losses, val_losses = [], []
    best_loss = float('inf')

    print("Training backtest model...")
    for ep in range(50):
        net.train()
        ep_loss = 0

        for cats, conts, y, _ in train_loader:
            opt.zero_grad()
            preds = net(cats, conts)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(y)

        train_losses.append(ep_loss / train_n)

        net.eval()
        v_loss = 0
        with torch.no_grad():
            for cats, conts, y, _ in val_loader:
                v_loss += loss_fn(net(cats, conts), y).item() * len(y)

        avg_v_loss = v_loss / val_n
        val_losses.append(avg_v_loss)

        if avg_v_loss < best_loss:
            best_loss = avg_v_loss
            torch.save(net.state_dict(), 'backtest_best.pth')

        if (ep + 1) % 10 == 0:
            print(
                f"Ep {ep + 1} | Train: {train_losses[-1]:.3f} | Val: {avg_v_loss:.3f}")

    # Load up the best weights before running the money sim
    net.load_state_dict(torch.load('backtest_best.pth'))

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.axvline(x=np.argmin(val_losses), color='r', linestyle='--',
                label='Best')
    plt.legend()
    plt.savefig('backtest_curve.png')

    run_sim(net, test_loader)


if __name__ == "__main__":
    main()