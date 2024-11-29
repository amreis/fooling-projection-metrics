import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch as T
import torch.linalg as linalg
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from tqdm import trange

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


class MetricLearner(nn.Module):
    def __init__(
        self, input_dim: int, proj_dim: int, out_dim:int=1, precision: float = 100.0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.out_dim = out_dim
        self.precision = precision

        self.precision_cell = nn.Parameter(
            T.ones((1,), dtype=T.float), requires_grad=True
        )
        self.mean_quality = nn.Sequential(
            nn.Linear(input_dim + proj_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Sigmoid(),
        )

        self.quality_1 = nn.Linear(input_dim + proj_dim, 512)
        self.quality_1_1 = nn.Linear(input_dim, 512)
        self.quality_1_2 = nn.Linear(proj_dim, 512)
        self.quality_2 = nn.Linear(512, 128)
        self.quality_3 = nn.Linear(128, 32)
        self.quality_4 = nn.Linear(32, out_dim)

        self.embedding_1 = nn.Linear(input_dim, 32)
        self.embedding_2 = nn.Linear(32, 2)

        self.embedding_x = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, inputs_high, inputs_proj):
        # q = F.relu(self.quality_1_1(inputs_high)) + F.relu(self.quality_1_2(inputs_proj))
        q = F.relu(self.quality_1(T.concat([inputs_high, inputs_proj], dim=1)))
        q = F.relu(self.quality_2(q))
        emb = F.relu(self.embedding_1(inputs_high))
        emb = F.sigmoid(self.embedding_2(emb))
        q = F.relu(self.quality_3(q))
        q = F.sigmoid(self.quality_4(q)).squeeze(dim=-1)

        # q = self.mean_quality(T.concat((inputs_high, inputs_proj), dim=1)).squeeze(
        #     dim=1
        # )

        # delta_q = 1 - T.exp(
        #     -(emb - inputs_proj).square().sum(dim=1) * self.precision_cell
        # )

        # return T.clip(q - delta_q, 0.0, 1.0)
        return q

    def init_params(self):
        self.apply(init_params)


def init_params(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)


class MetricNNP(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_dim = input_dim
        self.proj_dim = proj_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, proj_dim),
            nn.Sigmoid(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, inputs):
        proj = self.encoder(inputs)
        recon = F.sigmoid(self.decoder(proj))

        return proj, recon

    def project(self, inputs, grad=False):
        if not grad:
            with T.no_grad():
                return self.encoder(inputs)
        return self.encoder(inputs)

    def init_params(self):
        def _init_params(m: nn.Module):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 1e-4)

        self.apply(_init_params)


def main():
    from sklearn.preprocessing import minmax_scale

    if T.cuda.is_available():
        T.set_default_device("cuda")

    X = np.load(ROOT / "data" / "mnist" / "X.npy")
    X = minmax_scale(X).astype(np.float32)
    y = np.load(ROOT / "data" / "mnist" / "y.npy")

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=5000, stratify=y, random_state=420
    )

    X_proj = TSNE().fit_transform(X_train)
    X_proj = minmax_scale(X_proj)

    from neural_qmetrics.metrics import (
        continuity,
        per_point_continuity,
    )

    D_high = pdist(X_train)
    D_tsne = pdist(X_proj)
    tsne_trustworthiness = per_point_continuity(D_high, D_tsne)

    model = MetricLearner(X_train.shape[1], 2, precision=0.2)
    model.init_params()

    opt = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 100

    from torch.utils.data import TensorDataset, DataLoader

    ds = TensorDataset(
        T.tensor(X_train).float(),
        T.tensor(X_proj).float(),
        T.tensor(tsne_trustworthiness).float(),
    )
    dl = DataLoader(
        ds,
        batch_size=256,
        shuffle=True,
        generator=T.Generator("cuda" if T.cuda.is_available() else "cpu"),
    )

    loop = trange(1, epochs + 1)
    for e in loop:
        epoch_loss = 0.0
        epoch_n = 0
        for batch in dl:
            X_high_i, X_proj_i, Q_i = batch

            out_q = model(X_high_i, X_proj_i)
            # loss = T.exp(F.l1_loss(out_q, Q_i))
            # loss = T.exp(10*T.abs(out_q - Q_i)).mean()
            # loss = F.binary_cross_entropy(out_q, Q_i, weight=1 / (1e-6 + Q_i))
            loss = T.log(T.cosh(out_q - Q_i)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_n += out_q.size(0)
        loop.set_description(f"[epoch = {e}] Loss = {epoch_loss/epoch_n:.4f}")

    model.eval()

    nnp = MetricNNP(X_train.shape[1], 2)
    opt2 = optim.Adam(nnp.parameters(), lr=1e-3, maximize=True, weight_decay=0.005)

    ds_nnp = TensorDataset(T.tensor(X_train), T.tensor(X_proj))
    dl_nnp = DataLoader(
        ds_nnp,
        batch_size=256,
        generator=T.Generator("cuda" if T.cuda.is_available() else "cpu"),
    )

    from torch.utils.tensorboard import writer

    tb_writer = writer.SummaryWriter()
    loop = trange(1, 1500 + 1)
    proj_progress = {}
    for e in loop:
        epoch_loss = 0.0
        epoch_n = 0

        for batch in dl_nnp:
            X_high_i, X_proj_i = batch

            learned_proj, recon = nnp(X_high_i)
            # score = model(X_high_i, learned_proj).mean()
            # if (e // 20) % 2 == 0 and e < 500:
            if e < 5:
                score = -F.mse_loss(learned_proj, X_proj_i)
            else:
                score = model(X_high_i, learned_proj).mean()
            # score -= T.log1p(F.mse_loss(recon, X_high_i))
            # score -= T.log(T.cosh(recon - X_high_i)).mean()
            score -= F.binary_cross_entropy(recon, X_high_i)
            opt2.zero_grad()
            score.backward()
            opt2.step()

            epoch_loss -= score.item()
            epoch_n += learned_proj.size(0)
        loop.set_description(f"[epoch = {e}] Loss = {epoch_loss/epoch_n:.4f}")
        if e == 1 or e % 100 == 0:
            with T.no_grad():
                proj_progress[e] = (
                    nnp.project(T.tensor(X_train).float()).detach().cpu().numpy()
                )
                fig, ax = plt.subplots(
                    1, 1, subplot_kw={"aspect": "equal"}, figsize=(8, 8)
                )
                ax.scatter(*proj_progress[e].T, c=y_train)
                tb_writer.add_figure("projection", fig, global_step=e, close=True)
                tb_writer.add_scalar(
                    "proj_trustworthiness",
                    continuity(D_high, pdist(proj_progress[e])),
                    global_step=e,
                )

    nnp.eval()
    proj_final = nnp.project(T.tensor(X_train).float()).detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2)

    axes[0][0].scatter(*X_proj.T, c=y_train)

    axes[0][1].scatter(*proj_final.T, c=y_train)

    axes[1][0].hist(tsne_trustworthiness)
    with T.no_grad():
        predicted_trustworthiness = (
            model(T.tensor(X_train).float(), T.tensor(X_proj).float()).cpu().numpy()
        )

    axes[1][1].hist(predicted_trustworthiness)

    print(f"Trustworthiness of original proj: {tsne_trustworthiness.mean()}")
    print(f"Trustworthiness of learned proj: {continuity(D_high, pdist(proj_final))}")

    plt.show()

    fig, axes = plt.subplots(1, 5, subplot_kw={"aspect": "equal"})
    for ax, proj in zip(axes, proj_progress.values()):
        ax.scatter(*proj.T, c=y_train)
    axes[-1].scatter(*proj_final.T, c=y_train)
    plt.show()

    from matplotlib.colors import CenteredNorm

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"aspect": "equal"})
    ax1.scatter(
        *X_proj.T,
        c=predicted_trustworthiness - tsne_trustworthiness,
        cmap="coolwarm",
        norm=CenteredNorm(vcenter=0.0),
    )
    ax2.scatter(
        *X_proj.T,
        c=y_train,
        alpha=np.abs(predicted_trustworthiness - tsne_trustworthiness),
        cmap="tab10",
    )
    plt.show()


if __name__ == "__main__":
    main()
