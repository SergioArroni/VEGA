**Repositorio VEGA**

Este repositorio implementa el algoritmo genético VEGA (Vector Evaluated Genetic Algorithm), un algoritmo genético de optimización multiobjetivo diseñado para problemas con funciones objetivo vectoriales.

## Jerarquía de carpetas

El repositorio tiene la siguiente jerarquía de carpetas:

* `image`: Esta carpeta contiene imágenes generadas de los tests.
* `results`: Esta carpeta contiene los resultados de las ejecuciones del algoritmo VEGA.
* `VEGA`: Esta carpeta contiene el código fuente del algoritmo VEGA.
    * `binary`: Esta subcarpeta contiene el código fuente para el cruce y la mutación de cromosomas binarios.
    * `crossover`: Esta subcarpeta contiene el código fuente para los diferentes operadores de cruce.
    * `mutation`: Esta subcarpeta contiene el código fuente para los diferentes operadores de mutación.
    * `selection`: Esta subcarpeta contiene el código fuente para los diferentes operadores de selección.
    * `util`: Esta subcarpeta contiene funciones auxiliares.
* `main.py`: Este archivo contiene el código principal para ejecutar el algoritmo VEGA.

## Instalación

Para instalar el repositorio, siga estos pasos:

1. Clonar el repositorio:

```
git clone https://github.com/[usuario]/vega.git
```

2. Instalar las dependencias:

```
pip install -r requirements.txt
```

## Uso

Para ejecutar el algoritmo VEGA, siga estos pasos:

1. Modificar el archivo `main.py` para configurar los parámetros del algoritmo.
2. Ejecutar el archivo `main.py`:

```
python main.py
```

## Documentación

La documentación del algoritmo VEGA se encuentra en el archivo `docs/vega.md`.

## Contribuciones

Se aceptan contribuciones al repositorio. Para contribuir, siga estos pasos:

1. Forke el repositorio.
2. Haga los cambios necesarios en el código.
3. Envíe una solicitud de extracción.

## Licencia

El repositorio está licenciado bajo la licencia MIT.

## Referencias

* Schaffer, James David. "Some experiments in machine learning using vector evaluated genetic algorithms." (1985).
* C{\^\i}rciu, MIHAELA SIMONA, and FLORIN Leon. "Comparative study of multiobjective genetic algorithms." Bulletin of the Polytechnic Institute of Ia{\c{s}}i (2010): 35-47.
* Coello, Carlos A Coello. Evolutionary algorithms for solving multi-objective problems. Springer, 2007.
* Kursawe, Frank. "A variant of evolution strategies for vector optimization." International conference on parallel problem solving from nature (1990): 193-197.
* Hajela, Prabhat, and C -Y Lin. "Genetic search strategies in multicriterion optimal design." Structural optimization 4.2 (1992): 99-107.
* Zitzler, Eckart, and Lothar Thiele. "Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach." IEEE transactions on Evolutionary Computation 3.4 (1999): 257-271.
* Srinivas, Nidamarthi, and Kalyanmoy Deb. "Muiltiobjective optimization using nondominated sorting in genetic algorithms." Evolutionary computation 2.3 (1994): 221-248.
* Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary algorithms: Empirical results." Evolutionary computation 8.2 (2000): 173-195.

**Ejemplo de uso**

Para ejecutar el algoritmo VEGA para un problema con dos funciones objetivo, siga estos pasos:

1. Añada sus funciones como ZDT7 dentro de test/ZDT.py

2. Modifique el archivo `main.py` para configurar los parámetros del algoritmo. En este caso, los parámetros son:
    * `problem`: El tipo de problema a resolver.
    * `objective_functions`: Una lista de las funciones objetivo.
    * `population_size`: El tamaño de la población.
    * `number_of_iterations`: El número de iteraciones.

2. Ejecute el archivo `main.py`:

```
python main.py
```

El algoritmo VEGA ejecutará 100 iteraciones y generará una solución no dominada para el problema.

**Más información**

Para obtener más información sobre el algoritmo VEGA, consulte la documentación del repositorio.