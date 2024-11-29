# Fooling Projection Quality Metrics

This is the companion code repository to the work "Necessary but not Sufficient: Limitations of Projection Quality Metrics", by Alister Machado, Michael Behrisch, and Alexandru Telea (to appear, hopefully soon).

## Running the fooling pipeline

To run our code, we recommend you use [`uv`](https://github.com/astral-sh/uv).

```sh
uv venv
uv sync
source .venv/bin/activate

# Unzip the data to be able to run the experiments
cd data
chmod +x unzip.sh  # or $ bash unzip.sh
./unzip.sh

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

## About Me

My name is Alister Machado, I am a PhD Candidate researching Data Visualization (more specifically focused in dimensionality reduction and explainable AI) with the Visualization and Graphics Group (VIG) of Utrecht University, in the Netherlands. I am the person behind [ShaRP](https://github.com/amreis/sharp) and the [Differentiable DBMs](https://github.com/amreis/differentiable-dbm). You can check out my research [here](https://scholar.google.com.br/citations?user=WVXX6mYAAAAJ&hl=en). I am currently in the 4th year of my PhD (out of 5 total), and am expected to graduate around September 2026.
