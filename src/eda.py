import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(filepath: str):
    # Chargement
    df = pd.read_csv(filepath)
    print("=== BILAN SANTE DATASET ===")
    print(f"Lignes: {df.shape[0]}, Colonnes: {df.shape[1]}")
    print(f"Valeurs manquantes totales: {df.isnull().sum().sum()}")
    print("\nRépartition Cible (loan_status):")
    print(df["loan_status"].value_counts(normalize=True).round(4) * 100)

    # Comparaison des moyennes par statut de prêt
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != "loan_status"]

    mean_comparison = df.groupby("loan_status")[num_cols].mean().T
    mean_comparison.columns = ["Moyenne_Sain_0", "Moyenne_Defaut_1"]
    mean_comparison["Ecart_Relatif_%"] = (
        (mean_comparison["Moyenne_Defaut_1"] - mean_comparison["Moyenne_Sain_0"])
        / mean_comparison["Moyenne_Sain_0"]
        * 100
    ).round(2)

    print("\n=== COMPARAISON DES PROFILE RISK (0 vs 1) ===")
    print(mean_comparison)

    return df


if __name__ == "__main__":
    df = run_eda("data/raw/loan_default_prediction.csv")


def generate_correlation_mosaic(
    filepath: str, output_img: str = "correlation_mosaic.png"
):
    df = pd.read_csv(filepath)

    # Sélection des variables numériques
    num_cols = [
        "credit_score",
        "debt_to_income",
        "previous_defaults",
        "income",
        "employment_years",
        "age",
        "loan_amount",
        "interest_rate",
    ]

    # Configuration de la grille 2x2
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.suptitle(
        "Mosaïque d'Analyse des Corrélations & Risque", fontsize=16, fontweight="bold"
    )

    # 1. Matrice de corrélation globale
    sns.heatmap(
        df[num_cols + ["loan_status"]].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=axes[0, 0],
        cbar=False,
    )
    axes[0, 0].set_title("1. Corrélation Globale (Pearson)", fontweight="bold")

    # 2. Différence de corrélation (Défaut - Non Défaut)
    corr_sain = df[df["loan_status"] == 0][num_cols].corr()
    corr_defaut = df[df["loan_status"] == 1][num_cols].corr()
    diff_corr = corr_defaut - corr_sain

    sns.heatmap(
        diff_corr, annot=True, fmt=".2f", cmap="PiYG", ax=axes[0, 1], cbar=False
    )
    axes[0, 1].set_title(
        "2. Variation de Corrélation (Défaillants vs Sains)", fontweight="bold"
    )

    # 3. Focus sur Emprunts Personnels (Personal)
    corr_personal = df[df["loan_purpose"] == "Personal"][num_cols].corr()
    sns.heatmap(
        corr_personal, annot=True, fmt=".2f", cmap="Blues", ax=axes[1, 0], cbar=False
    )
    axes[1, 0].set_title("3. Corrélation - Motif: Personal", fontweight="bold")

    # 4. Focus sur Emprunts Professionnels (Business)
    corr_business = df[df["loan_purpose"] == "Business"][num_cols].corr()
    sns.heatmap(
        corr_business, annot=True, fmt=".2f", cmap="Greens", ax=axes[1, 1], cbar=False
    )
    axes[1, 1].set_title("4. Corrélation - Motif: Business", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_img, dpi=300, bbox_inches="tight")
    print(f"Mosaïque de corrélation sauvegardée sous : {output_img}")


if __name__ == "__main__":
    generate_correlation_mosaic("data/raw/loan_default_prediction.csv")
