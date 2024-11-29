import logging
from collections import ChainMap

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from tensorflow_projection_qm import metrics as tfpqm_metrics
from tensorflow_projection_qm.metrics import continuity, jaccard, neighborhood_hit, trustworthiness
from tensorflow_projection_qm.util import distance
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange
from umap import UMAP

import neural_qmetrics

ALLOWED_METRICS = ("trustworthiness", "continuity", "neighborhood_hit", "jaccard")


@tf.function
def trustworthiness_impl(X, X_2d, k) -> tf.Tensor:
    print("Tracing!")
    k = tf.cast(k, tf.int32)
    D_high = distance.psqdist(X)
    D_low = distance.psqdist(X_2d)

    n = tf.shape(D_high)[0]
    k = tf.minimum(k, n - 1)
    if 2 * k < n:
        norm_factor = k * (2 * n - 3 * k - 1) / 2
    else:
        norm_factor = (n - k) * (n - k - 1) / 2

    nn_orig = distance.sort_distances(D_high)
    ixs_orig = tf.argsort(nn_orig)

    knn_orig = nn_orig[:, 1 : k + 1]
    knn_proj = distance.nearest_k(D_low, k=k + 1)[1][:, 1:]

    U_i = tf.sparse.to_dense(tf.sets.difference(knn_proj, knn_orig), default_value=-1)
    pre_trust = tf.where(
        U_i >= 0, tf.gather(ixs_orig, tf.where(U_i >= 0, U_i, 0), batch_dims=-1) - k, 0
    )
    trust = tf.reduce_sum(pre_trust, -1)
    trust_t = tf.cast(trust, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)

    return tf.squeeze(1 - tf.math.divide_no_nan(trust_t, norm_factor))


@tf.function
def trustworthiness_from_dist(nn_orig, D_low, k) -> tf.Tensor:
    print("Tracing!")
    k = tf.cast(k, tf.int32)
    n = tf.shape(D_low)[0]
    k = tf.minimum(k, n - 1)
    if 2 * k < n:
        norm_factor = k * (2 * n - 3 * k - 1) / 2
    else:
        norm_factor = (n - k) * (n - k - 1) / 2

    ixs_orig = tf.argsort(nn_orig)

    knn_orig = nn_orig[:, 1 : k + 1]
    knn_proj = distance.nearest_k(D_low, k=k + 1)[1][:, 1:]

    U_i = tf.sparse.to_dense(tf.sets.difference(knn_proj, knn_orig), default_value=-1)
    pre_trust = tf.where(
        U_i >= 0, tf.gather(ixs_orig, tf.where(U_i >= 0, U_i, 0), batch_dims=-1) - k, 0
    )
    trust = tf.reduce_sum(pre_trust, -1)
    trust_t = tf.cast(trust, tf.float64)
    k = tf.cast(k, tf.float64)
    n = tf.cast(n, tf.float64)
    return tf.squeeze(1 - tf.math.divide_no_nan(trust_t, norm_factor))


ZADU_SPEC = [
    {"id": "tnc"},
    {"id": "mrre"},
    {"id": "lcmc"},
    {"id": "nh"},
    {"id": "nd"},
    {"id": "ca_tnc"},
    {"id": "proc"},
    {"id": "snc"},
    {"id": "dsc"},
    {"id": "ivm"},
    {"id": "c_evm"},
    {"id": "l_tnc"},
    {"id": "stress"},
    {"id": "kl_div"},
    {"id": "dtm"},
    # {"id": "topo"},  # metric is giving -inf quite often.
    {"id": "pr"},
    {"id": "srho"},
]

log = logging.getLogger(__name__)


def calculate_metric_per_point(X, X_proj, metric_name: str, k: int, y=None):
    assert metric_name in ALLOWED_METRICS

    if metric_name == "trustworthiness":
        return trustworthiness.trustworthiness_with_local(X, X_proj, k=k)[1].numpy()
    elif metric_name == "continuity":
        return continuity.continuity_with_local(X, X_proj, k=k)[1].numpy()
    elif metric_name == "neighborhood_hit":
        assert y is not None, "y must be provided for metric = neighborhood_hit"
        return neighborhood_hit.neighborhood_hit_with_local(X_proj, y, k=k)[1].numpy()
    else:  # metric_name == "jaccard":
        return jaccard.jaccard_with_local(X, X_proj, k=k)[1].numpy()


def calculate_all_metrics(X, X_proj, k: int, y=None, metric_calculator=None):
    if metric_calculator is None:
        # d = metrics.per_point_all_metrics(pdist(X), pdist(X_proj), y=y, k=k)
        n_classes = None if y is None else len(np.unique(y))
        d = tfpqm_metrics.run_all_metrics(
            X, X_proj.astype(X.dtype), y, k, n_classes=n_classes, as_numpy=True
        )
        return {k: v.mean() for k, v in d.items()}
    else:
        metric_vals = dict(ChainMap(*metric_calculator.measure(X_proj, y)))
        return metric_vals


def read_or_generate_projection(X: np.ndarray, dataset_name: str, proj_name: str):
    X_proj = neural_qmetrics.common.data.load_projection(dataset_name, proj_name)
    if X_proj is not None:
        log.info("Found previous projection")
        return X_proj
    log.info("Running projection algorithm...")
    match proj_name.lower().strip():
        case "tsne":
            X_proj = TSNE().fit_transform(X)
        case "umap":
            X_proj = UMAP().fit_transform(X)
        case "isomap":
            X_proj = Isomap().fit_transform(X)
        case "mds":
            X_proj = MDS().fit_transform(X)
        case _:
            raise ValueError(f"invalid projection name {proj_name}")
    X_proj = X_proj.astype(np.float32)
    log.info("saving projected output")
    neural_qmetrics.common.data.save_projection(dataset_name, proj_name, X_proj)
    log.info("done")
    return X_proj


def train_metric_learner(
    metric_learner: neural_qmetrics.models.metric_nnp.MetricLearner,
    X_high,
    X_proj,
    metric_values,
    epochs=100,
):
    ds = TensorDataset(
        T.tensor(X_high).float(),
        T.tensor(X_proj).float(),
        T.tensor(metric_values).float(),
    )
    dl = DataLoader(ds, batch_size=256, shuffle=True, generator=T.Generator(ds.tensors[0].device))

    opt = optim.Adam(metric_learner.parameters(), lr=1e-3)
    for e in (loop := trange(epochs)):
        epoch_loss = 0.0
        epoch_n = 0
        for X_high_i, X_proj_i, metric_i in dl:
            out_q = metric_learner(X_high_i, X_proj_i)
            loss = F.binary_cross_entropy(out_q, metric_i, weight=1 / (1e-6 + metric_i))
            # loss = T.log(T.cosh(out_q - metric_i)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            epoch_n += metric_i.size(0)
        loop.set_description(f"[epoch = {e}] Loss = {epoch_loss/epoch_n:.4f}")
    return metric_learner


def main():
    plt.rcParams.update(
        {
            "savefig.dpi": 256,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.0,
            "lines.markersize": 2.0,
        }
    )

    T.set_default_device(_device := "cuda" if T.cuda.is_available() else "cpu")
    log.info(f"using {_device} for PyTorch")
    dataset = "spambase"
    projection = "tsne"

    X, y = neural_qmetrics.common.data.load_dataset(dataset)
    print(f"{dataset}: {X.shape = }, {y.shape = }")
    log.info(f"Running for dataset = {dataset}, with {X.shape[1]} dimensions.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=min(5000, int(0.9 * X.shape[0])), stratify=y, random_state=420
    )
    X_train = minmax_scale(X_train.astype(np.float32))

    log.info(f"Projecting or reading previous {projection} projection")
    X_proj = read_or_generate_projection(X_train, dataset, projection)
    X_proj = minmax_scale(X_proj)

    X_train, _unused1, X_proj, _unused2 = train_test_split(
        X_train, X_proj, stratify=y_train, train_size=500, random_state=420
    )

    D_high = distance.psqdist(X_train)
    nn_orig = distance.sort_distances(D_high)

    trust_g, trust_i = trustworthiness.trustworthiness_with_local(X_train, X_proj, k=51)
    print(f"Initial Trust = {trust_g.numpy()}")
    metric_learner = neural_qmetrics.models.metric_nnp.MetricLearner(X.shape[1], 2)
    metric_learner.init_params()

    # First, take a point from the dataset. Any point will do.
    ix_to_change = 0
    resolution = 100
    X_proj_new = np.where(np.arange(X_proj.shape[0])[:, None] != ix_to_change, X_proj, np.nan)
    out = np.empty((resolution, resolution), dtype=np.float64)
    for i, x_move in enumerate(np.linspace(0.0, 1.0, resolution, endpoint=True)):
        print(i)
        for j, y_move in enumerate(np.linspace(0.0, 1.0, resolution, endpoint=True)):
            X_proj_new[ix_to_change, :] = np.array([x_move, y_move])
            D_low = distance.psqdist(X_proj_new)
            trust = tf.reduce_mean(
                trustworthiness_from_dist(nn_orig, D_low, k=tf.constant(51))
            ).numpy()
            out[resolution - j - 1, i] = trust

    train_metric_learner(metric_learner, X_train, X_proj, trust_i.numpy())
    out_approx = np.empty((resolution, resolution), dtype=np.float64)
    X_proj_new = np.where(np.arange(X_proj.shape[0])[:, None] != ix_to_change, X_proj, np.nan)
    with T.no_grad():
        X_train_t = T.tensor(X_train).float()
        X_proj_new_t = T.tensor(X_proj_new).float()
        for i, x_move in enumerate(np.linspace(0.0, 1.0, resolution, endpoint=True)):
            print(i)
            for j, y_move in enumerate(np.linspace(0.0, 1.0, resolution, endpoint=True)):
                X_proj_new_t[ix_to_change, :] = T.tensor([x_move, y_move]).float()
                trust = metric_learner(X_train_t, X_proj_new_t).mean().cpu().numpy()
                out_approx[resolution - j - 1, i] = trust

    np.save("outputs/approxqm/true.npy", out)
    np.save("outputs/approxqm/approx.npy", out_approx)

    fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    ax.imshow(out, cmap="viridis")
    fig.savefig("outputs/approxqm/True.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    ax.imshow(out_approx, cmap="viridis")
    fig.savefig("outputs/approxqm/Approx.png")
    plt.close(fig)

    plt.subplot(121)
    plt.imshow(out, cmap="viridis")
    plt.subplot(122)
    plt.imshow(out_approx, cmap="viridis")
    plt.show()


if __name__ == "__main__":
    main()
