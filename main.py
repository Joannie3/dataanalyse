import pandas as pd
#sert le calcule numerique (matrice) 
import numpy as np
#gestion de visuel du maths 
import matplotlib.pyplot as plt
import seaborn as sns
#outils pour le marchine lerning  tout les formules des reg linaire. train_test_split qui separt les jeux de données en 80% train et 20% test pour verifier qui ne triche pas
from sklearn.model_selection import train_test_split
#standardiser : x - moyenne x (center) / ecart type de x (centrée reduite )
from sklearn.preprocessing import StandardScaler
#class de logistic reg
from sklearn.linear_model import LogisticRegression
#score metrics
from sklearn.metrics import accuracy_score

#============ importation des données
path = "googleplaystore.csv"
raw_df = pd.read_csv(path, sep=",",decimal=".")
raw_df = raw_df.drop(['Current Ver', 'Android Ver'], axis=1)

#============ copie du dataset brut
jeuxvideos_df = raw_df
jeuxvideos_df.head()

#============ vérification des types
jeuxvideos_df.dtypes

#============ afficher la dimension
print(jeuxvideos_df.shape)

#============ importation des données
pathr = "googleplaystore_user_reviews.csv"
raw_dfr = pd.read_csv(pathr, sep=",")
raw_dfr = raw_dfr.drop(['Translated_Review'], axis=1)

#============ copie du dataset brut
jeuxvideos_dfr = raw_dfr
jeuxvideos_dfr.head()

#============ vérification des types
jeuxvideos_dfr.dtypes

#============ afficher la dimension
print(jeuxvideos_dfr.shape)

# Grouper les données par application et compter les occurrences de chaque sentiment
sentiment_counts = raw_dfr.groupby('App')['Sentiment'].value_counts().unstack().fillna(0)

# Compter le nombre de valeurs NaN par application
nan_counts = raw_dfr.groupby('App')['Sentiment'].apply(lambda x: x.isna().sum())

# Ajouter les valeurs NaN au DataFrame des comptages de sentiments
sentiment_counts['NaN'] = nan_counts

# Trouver la colonne avec la valeur la plus élevée par ligne
max_column = sentiment_counts.idxmax(axis=1)

# Ajouter le montant de la colonne 'NaN' à la colonne avec la valeur la plus élevée
for app, col in max_column.items():
    sentiment_counts.loc[app, col] += sentiment_counts.loc[app, 'NaN']

# Supprimer la colonne 'NaN'
sentiment_counts.drop(columns=['NaN'], inplace=True)

# Calculer la moyenne de la polarité du sentiment pour chaque application, en excluant les valeurs NaN
polarity_mean = raw_dfr.groupby('App')['Sentiment_Polarity'].mean()

# Calculer la moyenne de la subjectivité du sentiment pour chaque application, en excluant les valeurs NaN
subjectivity_mean = raw_dfr.groupby('App')['Sentiment_Subjectivity'].mean()

# Fusionner les deux DataFrames
result = pd.concat([sentiment_counts, polarity_mean, subjectivity_mean], axis=1)

# Renommer les colonnes
result.rename(columns={'Sentiment_Polarity': 'AVG_Pol', 'Sentiment_Subjectivity': 'AVG_Sub'}, inplace=True)

# Renommer les colonnes pour plus de clarté
# result.columns.name = 'Sentiment'

# Afficher le tableau résultant
print(result)


# Fusionner le DataFrame résultant avec le DataFrame de base en utilisant la colonne 'App'
merged_df = pd.merge(jeuxvideos_df, result, on='App', how='left')

# Remplacer la valeur "Everyone" dans la colonne 'Price' par 0
merged_df['Price'] = merged_df['Price'].replace('Everyone', 0)

# Enlever le symbole dollar "$" de la colonne 'Price'
merged_df['Price'] = merged_df['Price'].str.replace('$', '')

# Convertir la colonne 'Price' en type de données numérique
merged_df['Price'] = pd.to_numeric(merged_df['Price'])

# Remplacer la valeur "+" dans la colonne 'Installs' par 0
merged_df['Installs'] = merged_df['Installs'].str.replace('+', '')

# Remplacer la chaîne "Varies with device" par 0 dans la colonne 'Size'
merged_df['Size'] = merged_df['Size'].replace('Varies with device', '0k')

# Fonction pour convertir les valeurs en mégaoctets (Mo) en kilooctets (ko)
def convert_size(size):
    size = str(size)  # Assurer que size est une chaîne de caractères
    if 'M' in size or 'm' in size:
        return float(size.replace('M', '').replace('m', '')) * 1024
    elif 'K' in size or 'k' in size:
        return float(size.replace('K', '').replace('k', ''))
    else:
        return float(size)

# Appliquer la fonction de conversion à la colonne 'Size'
merged_df['Size'] = merged_df['Size'].apply(convert_size)

# Convertir la colonne 'Size' en type de données numériques
merged_df['Size'] = pd.to_numeric(merged_df['Size'], errors='coerce')

#enlever les valeurs = null par 0
merged_df.fillna(0, inplace=True)

# Enregistrer les données fusionnées dans un nouveau fichier
merged_df.to_csv("merged_googleplaystore.csv", index=False)

print(merged_df.head(55))

#on verifie les types de variables
print(merged_df.dtypes)