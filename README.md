# Fooling Projection Quality Metrics

This is the companion code repository to the work "Necessary but not Sufficient: Limitations of Projection Quality Metrics", by Alister Machado, Michael Behrisch, and Alexandru Telea (to appear, hopefully soon).

## Running the fooling pipeline

To run our code, we recommend you use [`uv`](https://github.com/astral-sh/uv).

```sh
uv venv
uv sync
source .venv/bin/activate
python experiments/1_learn_by_metric.py  # for example, to run 1st experiment.
```

## Data used for the manuscript

If you want to download the experimental data upon which we build the figures and tables on the paper, it is available at [OSF](https://osf.io/9n6fs/?view_only=ca0a2b34f04e4335961b9481f3101ada). Download the `tar.gz` file and extract it to the top-level directory of this repository:

```sh
tar xzf expdata.tar.gz
```

In doing so you will be able to run the notebooks that generate our paper's figures.

## Computing quality metrics, but fast

If you are interested in the fast computation of projection quality metrics, check out the package developed by me using Tensorflow for that exact purpose: [Github Repo](https://github.com/amreis/tf-projection-qm), [PyPI Package](https://pypi.org/p/tensorflow-projection-qm).


## Questions? Thoughts? Ideas?

Feel free to reach out to me via [email](mailto:a.machadodosreis@uu.nl).

