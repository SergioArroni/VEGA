**VEGA Repository**

This repository implements the VEGA (Vector Evaluated Genetic Algorithm), a multi-objective genetic algorithm designed for problems with vector objective functions.

## Folder hierarchy

The repository has the following folder hierarchy:

* `image`: This folder contains images generated from the tests.
* `results`: This folder contains the results of the VEGA algorithm executions.
* `VEGA`: This folder contains the source code of the VEGA algorithm.
    * `binary`: This subfolder contains the source code for crossover and mutation of binary chromosomes.
    * `crossover`: This subfolder contains the source code for the different crossover operators.
    * `mutation`: This subfolder contains the source code for the different mutation operators.
    * `selection`: This subfolder contains the source code for the different selection operators.
    * `util`: This subfolder contains auxiliary functions.
* `main.py`: This file contains the main code to run the VEGA algorithm.

## Installation

To install the repository, follow these steps:

1. Clone the repository:

```
git clone https://github.com/[user]/vega.git
```

2. Install the dependencies:

```
pip install -r requirements.txt
```

## Usage

To run the VEGA algorithm, follow these steps:

1. Modify the `main.py` file to configure the algorithm's parameters.
2. Run the `main.py` file:

```
python main.py
```

## Documentation

The documentation of the VEGA algorithm is located in the `docs/vega.md` file.

## Contributions

Contributions to the repository are welcome. To contribute, follow these steps:

1. Fork the repository.
2. Make the necessary changes to the code.
3. Submit a pull request.

## License

The repository is licensed under the MIT license.

## References

* Schaffer, James David. "Some experiments in machine learning using vector evaluated genetic algorithms." (1985).
* C{\^\i}rciu, MIHAELA SIMONA, and FLORIN Leon. "Comparative study of multiobjective genetic algorithms." Bulletin of the Polytechnic Institute of Ia{\c{s}}i (2010): 35-47.
* Coello, Carlos A Coello. Evolutionary algorithms for solving multi-objective problems. Springer, 2007.
* Kursawe, Frank. "A variant of evolution strategies for vector optimization." International conference on parallel problem solving from nature (1990): 193-197.
* Hajela, Prabhat, and C -Y Lin. "Genetic search strategies in multicriterion optimal design." Structural optimization 4.2 (1992): 99-107.
* Zitzler, Eckart, and Lothar Thiele. "Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach." IEEE transactions on Evolutionary Computation 3.4 (1999): 257-271.
* Srinivas, Nidamarthi, and Kalyanmoy Deb. "Muiltiobjective optimization using nondominated sorting in genetic algorithms." Evolutionary computation 2.3 (1994): 221-248.
* Zitzler, Eckart, Kalyanmoy Deb, and Lothar Thiele. "Comparison of multiobjective evolutionary algorithms: Empirical results." Evolutionary computation 8.2 (2000): 173-195.

**Example of use**

To run the VEGA algorithm for a problem with two objective functions, follow these steps:

1. Add your functions as ZDT7 inside test/ZDT.py

2. Modify the `main.py` file to configure the algorithm's parameters. In this case, the parameters are:
    * `problem`: The type of problem to solve.
    * `objective_functions`: A list of the objective functions.
    * `population_size`: The population size.
    * `number_of_iterations`: The number of iterations.

2. Run the `main.py` file:

```
python main.py
```

The VEGA algorithm will run 100 iterations and generate a non-dominated solution for the problem.

**More information**

For more information about the VEGA algorithm, please refer to the repository's documentation.