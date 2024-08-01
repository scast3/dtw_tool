# Pan American Energy Alineamiento de Pozos - Pasantia 2024

## Descripción

Este proyecto tiene como objetivo comparar y analizar perfiles de resistividad de pozos utilizando Dynamic Time Warping (DTW). Se utilizan varias estrategias para determinar ventanas adecuadas que ayuden al algoritmo encontrar las mejores correlaciones.

## Estructura del Proyecto

- **graph_setup.py**: Contiene la creación del grafo networkx `G` que se utiliza en el código.
- **calc_functions.py**: Contiene funciones de cálculo, incluyendo la implementación de `windows_test` y `optimal_window` que optimizan las ventanas para DTW.
- **marker_functions.py**: Contiene funciones relacionadas con la manipulación y análisis de marcadores en los pozos.
- **solve_functions.py**: Contiene funciones para recorrer el grafo
- **data/**: Directorio donde se almacenan los datos de entrada (perfiles de resistividad, marcadores, etc.).
- **old notebooks/**: Directorio donde se almacenan los primeros scripts que ya no se usan.
- **window_optimization.ipynb**: Encuentra la ventana optima del grafo y la usa en otros nodos.
- **markers_optimization.ipynb**: Crea subgrafos basados en los marcadores y corre el algoritmo.

## Librerías de Python

- Python 3.x
- `networkx`
- `numpy`
- `pandas`
- `matplotlib`
- `fastdtw`
- `scikit-learn`
- `hyperopt`

Se puede descargar usando `pip`:

```bash
pip install networkx numpy pandas matplotlib fastdtw scikit-learn hyperopt
```

## Información del Autor

Santiago Castillo

sacastillo2025@gmail.com

[linkedin](https://www.linkedin.com/in/santiagoalejandro-castillo/)

## Referencias

- [Wheeler Hale 2015 - Colorado School of Mines](https://repository.mines.edu/handle/11124/17145?show=full)
- [DTW well alignment - implementation of Wheeler Hale](https://github.com/ar4/wheeler_hale_2015)
