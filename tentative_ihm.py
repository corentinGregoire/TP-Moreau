import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def imc_category(row):
    """
    Calcule la catégorie de l'IMC (Indice de Masse Corporelle) en fonction de la taille et du poids d'une personne.

    Args:
        row (pandas Series): une ligne d'un DataFrame contenant les informations sur le poids et la taille d'une personne.

    Returns:
        str: la catégorie de l'IMC calculée, qui peut être l'une des suivantes:
            - Maigre: si l'IMC est inférieur à 18.5
            - Normal: si l'IMC est compris entre 18.5 et 25
            - Surpoids: si l'IMC est compris entre 25 et 30
            - Obèse: si l'IMC est supérieur à 30
    """
    taille_m = row['taille'] / 100  # Convertir la taille en mètres
    imc = row['Poids'] / (taille_m ** 2)  # Calculer l'IMC
    if imc < 18.5:
        return 'Maigre'
    elif imc < 25:
        return 'Normal'
    elif imc < 30:
        return 'Surpoids'
    else:
        return 'Obèse'


def transform_dataframe(data, single = False):
    """
    Transforme un DataFrame en appliquant diverses transformations telles que la création de nouvelles colonnes, la
    conversion de colonnes catégorielles en variables indicatrices, et la suppression de colonnes non nécessaires.

    Args:
        data (pandas DataFrame): le DataFrame à transformer.

    Returns:
        pandas DataFrame: le DataFrame transformé avec les modifications suivantes :
            - Ajout d'une colonne 'age_annee' représentant l'âge en années.
            - Création de 5 classes d'âge distinctes basées sur la colonne 'age_annee' et ajout d'une nouvelle colonne
              'classe_age'.
            - Conversion des colonnes 'malade', 'fumeur' et 'conso_alco' en variables indicatrices (0/1).
            - Ajout d'une nouvelle colonne 'IMC_Category' calculée à partir des colonnes 'Poids' et 'taille'.
            - Conversion des colonnes catégorielles en variables indicatrices en utilisant la fonction pandas
              get_dummies.
            - Suppression des colonnes non nécessaires 'classe_age_moins de 18 ans', 'genre_homme', 'age_annee', 'age',
              'Poids', 'taille', 'IMC'.
    """
    data["age_annee"] = round(data["age"] / 365.25, 2)
    bins = [0, 18, 30, 45, 60, 100]
    labels = ['moins de 18 ans', '18-30 ans', '30-45 ans', '45-60 ans', 'plus de 60 ans']
    data['classe_age'] = pd.cut(data['age_annee'], bins=bins, labels=labels)

    if not single:
        data['malade'] = data["malade"].map({'oui': 1, 'non': 0})

    data['fumeur'] = data["fumeur"].map({'fumeur': 1, 'non fumeur': 0})
    data['conso_alco'] = data["conso_alco"].map({'oui': 1, 'non': 0})
    data['IMC'] = data.apply(lambda row: row['Poids'] / ((row['taille'] / 100) ** 2), axis=1)
    data['IMC_Category'] = data.apply(imc_category, axis=1)

    cat_cols = data.select_dtypes(exclude=['int64', 'float64']).columns
    cat_cols = list(cat_cols)
    cat_cols.append("glycemie")
    df_onehot = pd.get_dummies(data, columns=cat_cols)
    df_onehot = df_onehot.drop(
        ['classe_age_moins de 18 ans', 'genre_homme', 'age_annee', 'age', 'Poids', 'taille', 'IMC'], axis=1)

    st.write(df_onehot.columns)

    return df_onehot


def plot_2D_data(X, Y, new_data=None):
    plt.scatter(X[Y == 0, 0], X[Y == 0, 1], label='Non malade', alpha=0.5)
    plt.scatter(X[Y == 1, 0], X[Y == 1, 1], label='Malade', alpha=0.5)

    if new_data is not None:
        plt.scatter(new_data[0], new_data[1], label='Nouvelle donnée', c='black', marker='x', s=100)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    st.pyplot(plt.gcf())


@st.cache_resource
def init():
    # Chargez les données et préparez les données pour l'ACP à 2 dimensions
    data = pd.read_csv('maladie_cardiaque.csv', usecols=lambda column: column != 'Unnamed: 0', index_col='id')
    df = transform_dataframe(data)

    st.write(df.columns)

    Y = df["malade"].values
    X = df.drop(['malade'], axis=1)

    # Instancier l'objet PCA avec le nombre de composantes principales souhaité
    pca = PCA(n_components=0.95)

    # Appliquer l'ACP sur les variables explicatives
    X_pca = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.33, random_state=42)

    # Sélectionner les deux variables les plus importantes pour chaque composante principale
    # var_1 = X.columns[pca.components_[0].argsort()[-2:][::-1]][0]
    # var_2 = X.columns[pca.components_[1].argsort()[-2:][::-1]][0]

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    # rfc.fit(X_pca, df["malade"])

    plot_2D_data(X_pca, Y)

    return rfc, X, Y, pca


def main():
    st.title("Prédiction de maladies cardiaques")

    # Afficher les données réduites à 2 dimensions
    st.header("Données réduites à 2 dimensions avec l'ACP")

    # Cached initialization of the graphic
    rfc, X, Y, pca = init()

    # Entrer de nouvelles données
    st.header("Entrer de nouvelles données")
    age = st.number_input("Age (jours)", min_value=0, max_value=100000, value=25000)
    genre = st.selectbox("Genre", ["homme", "femme"])
    taille = st.number_input("Taille (cm)", min_value=0, max_value=250, value=180)
    poids = st.number_input("Poids (kg)", min_value=0.0, max_value=300.0, value=80.0)
    pression_systo = st.number_input("Pression systolique", min_value=100, max_value=160, value=120)
    pression_diasto = st.number_input("Pression diastolique", min_value=60, max_value=100, value=80)
    cholesterol = st.selectbox("Cholesterol", ["normal", "eleve", "tres eleve"])
    glycemie = st.selectbox("Glycémie", [1, 2, 3])
    fumeur = st.selectbox("Fumeur", ["fumeur", "non fumeur"])
    conso_alco = st.selectbox("Consommation d'alcool", ["oui", "non"])
    activite_physique = st.selectbox("Activité physique", [1, 0])

    # Prédire la maladie cardiaque pour la nouvelle donnée
    if st.button("Prédire"):
        new_data = pd.DataFrame([[age, genre, taille, poids, pression_systo, pression_diasto, cholesterol, fumeur,
                                  conso_alco, glycemie, activite_physique]],
                                columns=['age', 'genre', 'taille', 'Poids', 'pression_systo', 'pression_diasto',
                                         'cholesterol', 'glycemie', 'fumeur', 'conso_alco', 'activite_physique'])
        new_data_transformed = transform_dataframe(new_data, single=True)
        new_data_transformed = new_data_transformed[X.columns]
        new_data_transformed_pca = pca.transform(new_data_transformed)
        prediction = rfc.predict(new_data_transformed_pca)
        prediction_proba = rfc.predict_proba(new_data_transformed_pca)

        # Afficher le résultat de la prédiction
        if prediction[0] == 0:
            st.write("Le patient n'est pas atteint d'une maladie cardiaque.")
        else:
            st.write("Le patient est atteint d'une maladie cardiaque.")

        st.write(f"Probabilité d'être malade : {prediction_proba[0][1] * 100:.2f}%")

        # Ajoutez le nouveau point sur le graphique
        plot_2D_data(X, Y, new_data=new_data_transformed[0])


if __name__ == "__main__":
    main()
