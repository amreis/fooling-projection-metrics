import logging
from collections import ChainMap
from functools import partial
from typing import Iterable

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.manifold import MDS, TSNE, Isomap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from tensorflow_projection_qm import metrics as tfpqm_metrics
from tensorflow_projection_qm.metrics import continuity, jaccard, neighborhood_hit, trustworthiness
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import trange
from umap import UMAP
from zadu import zadu

import neural_qmetrics
from neural_qmetrics.common import util

ALLOWED_METRICS = ("trustworthiness", "continuity", "neighborhood_hit", "jaccard")


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


def read_or_generate_projection(X: np.ndarray, dataset_name: str, proj_name: str, y=None):
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
    X_proj = minmax_scale(X_proj.astype(np.float32))
    fig = None
    if y is not None:
        fig, ax = setup_figure()
        ax.scatter(*X_proj.T, c=y, cmap="tab10", s=np.sqrt(20))
    log.info("saving projected output")
    neural_qmetrics.common.data.save_projection(dataset_name, proj_name, X_proj, fig=fig)
    if fig is not None:
        plt.close(fig)
    log.info("done")
    return X_proj


class Recorder:
    def __init__(
        self,
        metrics_: Iterable[str],
        X_original,
        y,
        k,
        use_tensorboard=True,
        metric_calculator=None,
    ):
        self.metrics = list(metrics_)  # potentially drop
        self.X_high = np.copy(X_original)
        self.D_high = None
        self.y = np.copy(y)
        self.n_classes = len(np.unique(self.y))
        self.k = k

        self.writer = SummaryWriter() if use_tensorboard else None
        self.zadu_inst = metric_calculator

        self.metrics_history: dict[int, dict[str, float]] = {}

        self.original_proj_metrics: dict[str, float] = {}

        self.metrics_runner = tfpqm_metrics.get_all_metrics_runner(
            {"k": self.k, "n_classes": self.n_classes}
        )

    def record(self, X_proj, global_step: int):
        metric_values = self._get_metric_values(X_proj)
        self.metrics_history[global_step] = dict(metric_values)
        if self.writer is not None:
            fig, ax = plt.subplots(1, 1, subplot_kw={"aspect": "equal"}, figsize=(8, 8))
            ax.scatter(*X_proj.T, c=self.y)
            self.writer.add_figure("projection", fig, global_step=global_step, close=True)

            self.writer.add_scalars("metric", metric_values, global_step=global_step)

    def to_csv(self, path: str):
        records = [({"epoch": e} | ms) for (e, ms) in self.metrics_history.items()]
        records.append({"epoch": -1} | self.original_proj_metrics)
        df = pd.DataFrame.from_records(records)
        if self.zadu_inst is not None:
            used_metrics = list(self.original_proj_metrics.keys())
        else:
            used_metrics = list(
                self.original_proj_metrics.keys() - {"epoch"}
            )  # list(ALLOWED_METRICS)
        df.to_csv(
            path,
            index=False,
            header=True,
            float_format="%.3f",
            columns=["epoch"] + sorted(used_metrics),
        )

    def record_original_proj_metrics(self, metrics_dict: dict[str, float]):
        self.original_proj_metrics = metrics_dict

    def _get_metric_values(self, X_proj):
        if self.zadu_inst:
            return dict(ChainMap(*self.zadu_inst.measure(X_proj, self.y)))
        # if self.D_high is None:
        #     self.D_high = pdist(self.X_high)
        ms = self.metrics_runner.measure_from_dict(
            {
                "X": self.X_high,
                "X_2d": X_proj.astype(self.X_high.dtype),
                "y": self.y,
            }
        )
        return {k: v.numpy().mean() for k, v in ms.items()}


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
    loop = trange(epochs)
    for e in loop:
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


def train_nnp(
    nnp: neural_qmetrics.models.metric_nnp.MetricNNP,
    metric_model: neural_qmetrics.models.metric_nnp.MetricLearner,
    X_high,
    X_proj,
    *,
    epochs=1500,
    recorder: Recorder | None = None,
):
    opt = optim.Adam(nnp.parameters(), lr=1e-3, maximize=True)
    ds = TensorDataset(T.tensor(X_high).float(), T.tensor(X_proj).float())
    dl = DataLoader(ds, batch_size=256, shuffle=True, generator=T.Generator(ds.tensors[0].device))

    loop = trange(1, epochs + 1)
    for e in loop:
        epoch_loss = 0.0
        epoch_n = 0

        for X_high_i, X_proj_i in dl:
            learned_proj, recon = nnp(X_high_i)
            score = metric_model(X_high_i, learned_proj).mean()
            if e < 10:
                score -= 10 * F.mse_loss(learned_proj, X_proj_i)
            score -= T.log1p(F.mse_loss(recon, X_high_i))

            # if (e // 20) % 2 == 0 and e < 100:
            #     score = -F.mse_loss(learned_proj, X_proj_i)
            # else:
            #     score = metric_model(X_high_i, learned_proj).mean() - 0.01 * F.mse_loss(
            #         learned_proj, X_proj_i
            #     )
            opt.zero_grad()
            score.backward()
            opt.step()

            epoch_loss -= score.item()
            epoch_n += learned_proj.size(0)
        loop.set_description(f"[epoch = {e}] Loss = {epoch_loss/epoch_n:.4f}")
        if recorder is not None and (e == 1 or e % 100 == 0):
            proj_intermediate = nnp.project(T.tensor(X_high).float()).detach().cpu().numpy()
            recorder.record(proj_intermediate, global_step=e)


def setup_figure():
    fig, ax = plt.subplots(1, 1, subplot_kw={"aspect": "equal"}, figsize=(8, 8), dpi=256)
    ax.axis("off")
    ax.set_autoscale_on(False)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    return fig, ax


def plot_and_record_metrics(
    X_proj: np.ndarray,
    X_high: np.ndarray,
    y: np.ndarray,
    *,
    k: int,
    figname: str,
    method_name: str,
    out_metrics: list[dict],
    metric_calculator: zadu.ZADU | None = None,
):
    fig, ax = setup_figure()
    X_proj = util.rescale_projection_plot(X_proj.copy())
    ax.scatter(*X_proj.T, c=y)
    fig.savefig(figname)
    out_metrics.append(
        {"method": method_name}
        | calculate_all_metrics(X_high, X_proj, k=k, y=y, metric_calculator=metric_calculator)
    )
    plt.close(fig)


def make_filter_many(*args: np.ndarray):
    """Function to apply a single filter `b_ixs` below to many ndarrays."""

    def fn(b_ixs: np.ndarray):
        return tuple(arg[b_ixs] for arg in args)

    return fn


def postprocess_and_save_projections(
    X_proj: np.ndarray, X_high: np.ndarray, y: np.ndarray, *, k: int, metric_calculator=None
) -> None:
    log.info("Postprocessing obtained projection and calculating metrics")
    X_proj_rescaled = util.rescale_projection_plot(X_proj)

    metric_records: list[dict] = []
    plot_and_record = partial(
        plot_and_record_metrics,
        k=k,
        out_metrics=metric_records,
        metric_calculator=metric_calculator,
    )

    plot_and_record(
        X_proj_rescaled,
        X_high,
        y,
        figname="./Learned_Proj_Rescaled.png",
        method_name="rescaled",
    )

    filter_data = make_filter_many(X_proj, X_high, y)

    # TODO: uncomment.
    # inliers_90p = util.find_inliers_percentile(X_proj, p=0.9)
    # plot_and_record(
    #     *filter_data(inliers_90p),
    #     figname="./Learned_Proj_Inliers_90p.png",
    #     method_name="inliers_90p",
    # )

    # inliers_hdbscan = util.find_inliers_hdbscan(X_proj)
    # plot_and_record(
    #     *filter_data(inliers_hdbscan),
    #     figname="./Learned_Proj_Inliers_HDBSCAN.png",
    #     method_name="inliers_hdbscan",
    # )

    # inliers_lof = util.find_inliers_lof(X_proj)
    # plot_and_record(
    #     *filter_data(inliers_lof),
    #     figname="./Learned_Proj_Inliers_LOF.png",
    #     method_name="inliers_lof",
    # )

    p_magnetic = util.push_magnetic_repulsion(X_proj, n_iter=1, k=7, jitter=False)
    plot_and_record(
        p_magnetic,
        X_high,
        y,
        figname="./Learned_Proj_Repulsion.png",
        method_name="magnetic",
    )

    p_rescaled_magnetic = util.push_magnetic_repulsion(X_proj_rescaled, n_iter=1, k=7, jitter=False)
    plot_and_record(
        p_rescaled_magnetic,
        X_high,
        y,
        figname="./Learned_Proj_Rescaled_Repulsion.png",
        method_name="magnetic_on_rescale",
    )

    p_delaunay = util.force_directed_delaunay(X_proj_rescaled, n_iter=5)
    plot_and_record(
        p_delaunay,
        X_high,
        y,
        figname="./Learned_Proj_Delaunay.png",
        method_name="delaunay",
    )

    p_projnn = util.force_directed_projnn(X_proj_rescaled, n_iter=5)
    plot_and_record(p_projnn, X_high, y, figname="./Learned_Proj_ProjNN.png", method_name="projnn")

    p_truenn = util.force_directed_truenn(X_proj_rescaled, X_high, n_iter=5)
    plot_and_record(p_truenn, X_high, y, figname="./Learned_Proj_TrueNN.png", method_name="truenn")

    log.info("Saving results to CSV")
    df = pd.DataFrame.from_records(metric_records)
    df.to_csv(
        "./postprocess_metrics.csv",
        index=False,
        header=True,
        float_format="%.3f",
        columns=["method"] + sorted(list(metric_records[0].keys() - {"method"})),
    )
    log.info("Done!")


@hydra.main(version_base=None, config_name="exp1", config_path="conf")
def main(cfg: DictConfig):
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

    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset

    X, y = neural_qmetrics.common.data.load_dataset(dataset)
    print(f"{dataset}: {X.shape = }, {y.shape = }")
    log.info(f"Running for dataset = {dataset}, with {X.shape[1]} dimensions.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=min(5000, int(0.9 * X.shape[0])), stratify=y, random_state=420
    )
    X_train = minmax_scale(X_train.astype(np.float32))

    metric_calculator = zadu.ZADU(ZADU_SPEC, X_train) if cfg.use_zadu else None

    metric_learner = neural_qmetrics.models.metric_nnp.MetricLearner(X.shape[1], 2)
    metric_learner.init_params()
    log.info(f"Projecting or reading previous {cfg.projection} projection")
    X_proj = read_or_generate_projection(X_train, cfg.dataset, cfg.projection, y=y_train)
    log.info("Calculating metrics...")
    original_proj_metrics = calculate_all_metrics(
        X_train, X_proj, k=cfg.k, y=y_train, metric_calculator=metric_calculator
    )
    print(original_proj_metrics)
    log.info("Done.")

    log.info(f"Calculating metric to be optimized, {cfg.metric}, using k={cfg.k}")
    metric = calculate_metric_per_point(X_train, X_proj, cfg.metric, k=cfg.k, y=y_train)
    log.info("Done")

    train_metric_learner(metric_learner, X_train, X_proj, metric)
    metric_learner.eval()
    with T.no_grad():
        metric_learner_output = (
            metric_learner(T.tensor(X_train).float(), T.tensor(X_proj).float()).cpu().numpy()
        )
        err_metric_learner = ((metric_learner_output - metric) ** 2).mean()
    log.info(f"Error of metric learner predictions: {err_metric_learner:.5f}")

    nnp = neural_qmetrics.models.metric_nnp.MetricNNP(X_train.shape[1], 2)
    nnp.init_params()
    recorder = Recorder(
        ALLOWED_METRICS,
        X_train,
        y_train,
        k=cfg.k,
        use_tensorboard=cfg.use_tensorboard,
        metric_calculator=metric_calculator,
    )
    log.info("Starting NNP training...")
    train_nnp(nnp, metric_learner, X_train, X_proj, recorder=recorder, epochs=cfg.epochs)
    log.info("Done")
    recorder.record_original_proj_metrics(original_proj_metrics)
    log.info("Saving metrics per epoch to metrics_per_epoch.csv")
    recorder.to_csv("./metrics_per_epoch.csv")

    with plt.rc_context(rc={"lines.markersize": np.sqrt(20), "image.cmap": "tab10"}):
        log.info("Creating resulting projection")
        learned_proj = nnp.project(T.tensor(X_train).float()).cpu().numpy()
        np.save("./Learned_Proj.npy", learned_proj)
        fig, ax = setup_figure()
        ax.scatter(*learned_proj.T, c=y_train)
        log.info("Saving it to Learned_Proj.png")
        fig.savefig("./Learned_Proj.png")
        plt.close(fig)
        postprocess_and_save_projections(
            learned_proj, X_train, y_train, k=cfg.k, metric_calculator=metric_calculator
        )

    log.info("Finalizing...")


if __name__ == "__main__":
    main()
