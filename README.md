# Pan American Energy Well Alignment Model

## Overview

**PAE Well Alignment Model** is a Python-based tool for analyzing and predicting depth data in geological surveys using resistivity measurements. The project leverages NetworkX for graph operations, Pandas for data manipulation, and various statistical and machine learning methods for prediction.

## Features

- Analyze depth data from geological surveys.
- Predict geological tops based on resistivity data.
- Compare predictions with actual known tops.
- Optimize prediction windows using Hyperopt.
- Visualize data and results with Matplotlib.

## Requirements

- Python 3.x
- `networkx`
- `numpy`
- `pandas`
- `matplotlib`
- `fastdtw`
- `scikit-learn`
- `hyperopt`

You can install the required packages using `pip`:

```bash
pip install networkx numpy pandas matplotlib fastdtw scikit-learn hyperopt
```

## Author Information

Santiago Castillo

sacastillo2025@gmail.com

[linkedin](https://www.linkedin.com/in/santiagoalejandro-castillo/)

## Acknowledgments

This project took inspiration from existing work:

- [Wheeler Hale 2015 - Colorado School of Mines](https://repository.mines.edu/handle/11124/17145?show=full)
- [DTW well alignment - implementation of Wheeler Hale](https://github.com/ar4/wheeler_hale_2015)
- [Hyperopt optimization](https://hyperopt.github.io/hyperopt/)
