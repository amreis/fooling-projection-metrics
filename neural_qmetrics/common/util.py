"""Module with utilities for data manipulation in experiments.

This module has:
* Functions for post-processing projection plots (finding outliers,
rescaling, etc.).
"""

import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


def rescale_projection_plot(X_proj: np.ndarray) -> np.ndarray:
    """Rescale projection so that it lies in [0, 1] for all axes."""
    mins = np.min(X_proj, axis=0)
    maxs = np.max(X_proj, axis=0)

    return (X_proj - mins) / (maxs - mins)


def find_inliers_percentile(X_proj: np.ndarray, *, p=0.95) -> npt.NDArray[np.int_]:
    """
    Drop points too far away from the centroid of the projection.

    Drops points whose distance to the centroid of the projection is
    in the p'th percentile. If p == 1, drops nothing. If p == 0, drops
    everything.
    """

    centroid = np.expand_dims(X_proj.mean(axis=0), axis=0)
    distances = np.linalg.norm(centroid - X_proj, axis=1)

    n = X_proj.shape[0]
    pn = int(p * n)
    keep_indices = np.argsort(distances)[:pn]

    return keep_indices


def find_inliers_isolation_forest(X_proj: np.ndarray) -> npt.NDArray[np.bool_]:
    ifor = IsolationForest()
    ifor.fit(X_proj)

    results = ifor.predict(X_proj)

    return results == 1


def find_inliers_hdbscan(
    X_proj: np.ndarray, *, dist_threshold: float | None = None
) -> npt.NDArray[np.bool_]:
    dists = pdist(X_proj)

    hdbscan = HDBSCAN(allow_single_cluster=True)
    hdbscan.fit(X_proj)

    if dist_threshold is None:
        dist_threshold = np.median(dists)

    return hdbscan.dbscan_clustering(dist_threshold) != -1


def find_inliers_lof(X_proj: np.ndarray, *, k=20) -> npt.NDArray[np.bool_]:
    lof = LocalOutlierFactor(n_neighbors=k)
    is_outlier = lof.fit_predict(X_proj)

    return is_outlier != -1


def push_towards_lower_stress(
    X_proj: np.ndarray,
    X_high: np.ndarray,
    *,
    k=7,
    n_iter=20,
) -> np.ndarray:
    X_proj = np.copy(X_proj)
    X_high = np.copy(X_high)  # just to be safe

    D_high = squareform(pdist(X_high, metric="sqeuclidean"))

    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(X_high)

    neighbors = nn.kneighbors(X_high, return_distance=False)
    neighbors = neighbors[:, 1 : k + 1]
    for _ in range(n_iter):
        D_low = squareform(pdist(X_proj, metric="sqeuclidean"))
        for i in range(X_high.shape[0]):
            low_distances = D_low[i, neighbors[i]]
            high_distances = D_high[i, neighbors[i]]

            diffs = high_distances - low_distances
            diffs[diffs < 0] = diffs[diffs < 0] * 0.1
            diffs[diffs > 1.0] = 1.0

            f = np.zeros_like(X_proj[0])
            for j, mag in zip(neighbors[i], diffs):
                direction = (X_proj[j] - X_proj[i]) / (1e-6 + np.linalg.norm(X_proj[j] - X_proj[i]))
                f += direction * mag

            X_proj[i] += 0.01 * f

    return X_proj


def push_magnetic_repulsion(X_proj: np.ndarray, *, k=5, n_iter=20, jitter=False) -> np.ndarray:
    X_proj = np.copy(X_proj)

    def _force_log_spring(ds: np.ndarray, *, ideal_dist=0.01) -> npt.NDArray[np.float32]:
        return np.log(ideal_dist / ds**2)

    def _force_bell(ds: np.ndarray, precision=1.0) -> npt.NDArray[np.float32]:
        return np.exp(-(precision * ds**2))

    lr = 0.1

    if jitter:
        X_proj += np.random.randn(*X_proj.shape) * 0.001

    for e in range(n_iter):
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_proj)
        distances, neighbor_ixs = nn.kneighbors(X_proj)

        neighbor_ixs = neighbor_ixs[:, 1:]  # remove self-distance (== 0)
        distances = distances[:, 1:]

        f = np.zeros_like(X_proj)
        for i in range(X_proj.shape[0]):
            ns = neighbor_ixs[i]
            ds = distances[i]

            neighbors = X_proj[ns]
            directions = X_proj[i] - neighbors
            directions /= np.expand_dims(ds, 1) + np.finfo(float).eps

            # np.where(ds <= 0.5, 0.01 * np.log(0.01 / ds**2), 0.0)

            f[i] = np.sum(
                np.expand_dims(0.1 * _force_bell(ds, precision=0.5), 1) * directions,
                axis=0,
            )
            f[i] += np.sum(np.expand_dims(np.where(ds < 0.1, -0.01, 0.0), 1) * directions, axis=0)
        # X_proj += 0.01 * f

        # X_proj += (0.1 / (1 + 5 * e)) * f

        X_proj += lr * f
        X_proj = rescale_projection_plot(X_proj)
        if np.max(np.linalg.norm(lr * f, axis=1)) <= 1e-5:
            print(f"forces converged in epoch {e}, terminating")
            break
        lr *= 0.8
    return X_proj


def force_directed_delaunay(X_proj: np.ndarray, *, n_iter=50) -> np.ndarray:
    from scipy.spatial import Delaunay

    proj_distances = pdist(X_proj)
    if np.any(np.isclose(proj_distances, 0.0)):
        gen = np.random.default_rng()
        X_proj += 0.001 * gen.uniform(-1.0, 1.0, size=X_proj.shape)
    min_dist = np.maximum(np.percentile(proj_distances, q=0.4), 1e-6)

    dt = Delaunay(X_proj, qhull_options="QJ")

    indptr, indices = dt.vertex_neighbor_vertices

    adj_list = {k: indices[indptr[k] : indptr[k + 1]] for k in range(X_proj.shape[0])}

    G = nx.Graph(adj_list)
    layout = nx.spring_layout(
        G,
        k=min_dist,
        pos={k: X_proj[k] for k in range(X_proj.shape[0])},
        iterations=n_iter,
    )
    return np.array([layout[k] for k in range(X_proj.shape[0])])


def force_directed_projnn(X_proj: np.ndarray, *, k: int = 7, n_iter=50) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_proj)
    proj_distances = pdist(X_proj)
    if np.any(np.isclose(proj_distances, 0.0)):
        gen = np.random.default_rng()
        X_proj += 0.001 * gen.uniform(-1.0, 1.0, size=X_proj.shape)

    G = nx.Graph(nn.kneighbors_graph(mode="connectivity"))

    layout = nx.spring_layout(
        G, pos={k: X_proj[k] for k in range(X_proj.shape[0])}, iterations=n_iter
    )

    return np.array([layout[k] for k in range(X_proj.shape[0])])


def force_directed_truenn(
    X_proj: np.ndarray,
    X_high: np.ndarray,
    *,
    k: int = 7,
    n_iter=50,
) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X_high)
    proj_distances = pdist(X_proj)
    if np.any(np.isclose(proj_distances, 0.0)):
        gen = np.random.default_rng()
        X_proj += 0.001 * gen.uniform(-1.0, 1.0, size=X_proj.shape)

    G = nx.Graph(nn.kneighbors_graph(mode="distance"))
    layout = nx.spring_layout(
        G, pos={k: X_proj[k] for k in range(X_proj.shape[0])}, iterations=n_iter
    )
    return np.array([layout[k] for k in range(X_proj.shape[0])])
