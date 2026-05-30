import os
import random
from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score


def ustaw_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.random.set_seed(seed)


def wykres_krzywych_uczenia(history, metryki: Iterable[str] = ("loss", "accuracy"), tytul: str = "") -> None:
    hist = history.history
    n = len(list(metryki))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, metryka in zip(axes, metryki):
        if metryka in hist:
            ax.plot(hist[metryka], label=f"train {metryka}")
        val_klucz = f"val_{metryka}"
        if val_klucz in hist:
            ax.plot(hist[val_klucz], label=f"val {metryka}")
        ax.set_xlabel("Epoka")
        ax.set_ylabel(metryka)
        ax.set_title(f"{tytul} - {metryka}".strip(" -"))
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def macierz_pomylek(y_prawdziwe, y_predykcja, etykiety=None, tytul: str = "Macierz pomyłek") -> None:
    cm = confusion_matrix(y_prawdziwe, y_predykcja)
    plt.figure(figsize=(8, 6))
    etykiety = etykiety if etykiety is not None else "auto"
    annot = cm.shape[0] <= 15
    sns.heatmap(cm, annot=annot, fmt="d", cmap="Blues", xticklabels=etykiety, yticklabels=etykiety)
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista klasa")
    plt.title(tytul)
    plt.tight_layout()
    plt.show()


def raport_klasyfikacji(y_prawdziwe, y_predykcja, etykiety=None) -> str:
    return classification_report(y_prawdziwe, y_predykcja, target_names=etykiety, zero_division=0)


def metryki_regresji(y_prawdziwe, y_predykcja) -> dict:
    mse = mean_squared_error(y_prawdziwe, y_predykcja)
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_prawdziwe, y_predykcja),
        "R2": r2_score(y_prawdziwe, y_predykcja),
    }


def wykres_predykcja_vs_prawdziwa(y_prawdziwe, y_predykcja, tytul: str = "Predykcja vs wartość rzeczywista") -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_prawdziwe, y_predykcja, alpha=0.4, s=15)
    mn = float(min(np.min(y_prawdziwe), np.min(y_predykcja)))
    mx = float(max(np.max(y_prawdziwe), np.max(y_predykcja)))
    plt.plot([mn, mx], [mn, mx], "r--", label="y = x (idealna predykcja)")
    plt.xlabel("Wartość rzeczywista")
    plt.ylabel("Predykcja modelu")
    plt.title(tytul)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def tabela_porownawcza(wyniki: dict) -> "pd.DataFrame":
    import pandas as pd
    return pd.DataFrame(wyniki).T
