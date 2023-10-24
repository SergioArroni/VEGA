from pymoo.indicators.hv import HV
from pymoo.indicators.gd_plus import GDPlus
from pymoo.indicators.igd_plus import IGDPlus
import matplotlib.pyplot as plt
from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem
import seaborn as sns
import numpy as np


# Función de selección para encontrar soluciones no dominadas en el espacio de Pareto
def find_nondominated(objective_functions_values):
    n = len(objective_functions_values)
    nondominated_solutions = []

    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i != j:  # No comparamos la solución con ella misma
                if np.all(
                    objective_functions_values[i] <= objective_functions_values[j]
                ) and np.any(
                    objective_functions_values[i] < objective_functions_values[j]
                ):
                    is_dominated = True
                    break  # La solución i está dominada por al menos una solución

        if not is_dominated:
            nondominated_solutions.append(i)

    return nondominated_solutions


# Función para graficar las soluciones no dominadas
def plot_nondominated_solutions(x_data, y_data):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_data, y_data, c="b", marker="o", label="Soluciones no dominadas")
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("Función Objetivo 1")
    plt.ylabel("Función Objetivo 2")
    plt.title("Soluciones no Dominadas en el Espacio de Pareto")
    plt.legend()
    plt.grid(True)
    plt.show()


def imprimir(ZDT, results, pf):
    # Imprime las soluciones no dominadas
    a = open(f"../results/resultados_{ZDT}_P.txt", "a")
    medias = {}
    medias["f1"] = 0
    medias["f2"] = 0
    a.write(f"Soluciones no dominadas encontradas: {ZDT}\n")
    a.write(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    )
    for i, _ in enumerate(results["nondominated_solutions"]):
        a.write(
            f"{i + 1}, Funcion 1: {results['values'][i][0]}, Funcion 2: {results['values'][i][1]}\n"
        )

        medias["f1"] += results["values"][i][0]
        medias["f2"] += results["values"][i][1]
    a.write(
        "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n"
    )
    a.write(f"Elementos: {len(results['nondominated_solutions'])}\n")
    a.write(
        f"Media Funcion 1: {medias['f1'] / len(results['nondominated_solutions'])}\n"
    )
    a.write(
        f"Media Funcion 2: {medias['f2'] / len(results['nondominated_solutions'])}\n"
    )
    a.write(f"Hipervolumen: {hypervolume(results)}\n")
    a.write(f"GDPlus: {GDPlus_Value(pf, results)}\n")
    a.write(f"IGDPlus: {IGDPlus_Value(pf, results)}\n")

    a.write("------------------------------------------------------------------\n")
    a.close()


def hypervolume(results):
    ref_point = np.array([1.1, 1.1])
    ind = HV(ref_point=ref_point)
    return ind(results["values"])


def GDPlus_Value(pf, results):
    ind = GDPlus(pf)
    return ind(results["values"])


def IGDPlus_Value(pf, results):
    ind = IGDPlus(pf)
    return ind(results["values"])


def bind_parameters(func, **kwargs):
    def wrapped(chromosome):
        return func(chromosome, **kwargs)

    return wrapped


def plot_Scatter(pf, results, ZDT):
    # plot the result
    image = Scatter(legend=True)
    image.add(pf, label="Pareto-front")
    image.add(results["values"], label="Result")
    image.save(f"../image/generados/vega_{ZDT}_P.png")


def normalizar(results):
    # Encuentra el valor mínimo y máximo en cada columna
    min_column0 = np.min(results["values"][:, 0])
    max_column0 = np.max(results["values"][:, 0])
    min_column1 = np.min(results["values"][:, 1])
    max_column1 = np.max(results["values"][:, 1])

    # Normaliza los valores en ambas columnas
    results["values"][:, 0] = (results["values"][:, 0] - min_column0) / (
        max_column0 - min_column0
    )
    results["values"][:, 1] = (results["values"][:, 1] - min_column1) / (
        max_column1 - min_column1
    )
    return results


def regresion(results):
    # create scatterplot with regression line
    sns.regplot(
        x=results["values"][:, 0],
        y=results["values"][:, 1],
        lowess=True,
        ci=99,
        marker="o",
        color=".3",
        line_kws=dict(color="r"),
    )
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.xlabel("Función Objetivo 1")
    plt.ylabel("Función Objetivo 2")
    plt.show()


def get_pf(ZDT):
    return get_problem(str(ZDT)).pareto_front()
