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

#==================================================================================#
#==============Je calcul le nombre de fois ou le sentiment apparait ===============#
#==================================================================================#

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



#============================================================================================================#
#==============Je fais une moyenne pour le sentiment polarity je ne prends pas en compte Nan ===============#
#============================================================================================================#

# Calculer la moyenne de la polarité du sentiment pour chaque application, en excluant les valeurs NaN
polarity_mean = raw_dfr.groupby('App')['Sentiment_Polarity'].mean()

# Calculer la moyenne de la subjectivité du sentiment pour chaque application, en excluant les valeurs NaN
subjectivity_mean = raw_dfr.groupby('App')['Sentiment_Subjectivity'].mean()

# Fusionner les deux DataFrames
result = pd.concat([sentiment_counts, polarity_mean, subjectivity_mean], axis=1)

# Renommer les colonnes
result.rename(columns={'Sentiment_Polarity': 'AVG_Pol', 'Sentiment_Subjectivity': 'AVG_Sub'}, inplace=True)

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

#================================================================================#
#============== On converti l'ensemble de la colonne en KiloOctet ===============#
#================================================================================#

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

#==========================================================================#
#==============Ici on supprime les + dans la partie install ===============#
#==========================================================================#

merged_df['Installs'] = merged_df['Installs'].str.replace(',', '')

# Convertir les valeurs en entiers
merged_df['Installs'] = merged_df['Installs'].astype(int)

#===============================================================================================================#
#==============On calcul la moyenne de Rating sans prendre en compte les 0 et selon la catégorie ===============#
#===============================================================================================================#


# Filtrer les lignes avec une note différente de zéro
filtered_merged_df = merged_df[merged_df['Rating'] != 0.0]

# Calculer la moyenne de la colonne Rating par catégorie après avoir filtré les données
rating_mean_by_category = filtered_merged_df.groupby('Category')['Rating'].mean()

# Créer un dictionnaire contenant les moyennes de rating par catégorie
rating_mean_dict = rating_mean_by_category.to_dict()

# Parcourir le DataFrame et remplacer les valeurs nulles dans la colonne Rating
# par la moyenne correspondante de la catégorie, en arrondissant à 1 chiffre après la virgule
for index, row in merged_df.iterrows():
    if row['Rating'] == 0.0:
        category_mean = rating_mean_dict.get(row['Category'], 0.0)
        merged_df.loc[index, 'Rating'] = round(category_mean, 1)

#======================================================================================================#
#==============Je fais une moyenne pour les reviews et je remplace ou la valeur est a 0 ===============#
#======================================================================================================#


# Convertir la colonne Reviews en numérique en remplaçant les valeurs non numériques par NaN
merged_df['Reviews'] = pd.to_numeric(merged_df['Reviews'], errors='coerce')

# Calculer la moyenne des Reviews en excluant les lignes avec 0
reviews_mean = merged_df[merged_df['Reviews'] != 0]['Reviews'].mean()

# Remplacer les valeurs 0 dans la colonne Reviews par la moyenne calculée
merged_df['Reviews'] = merged_df['Reviews'].replace(0, reviews_mean)

# Filtrer les lignes avec une valeur Size différente de zéro
filtered_merged_df = merged_df[merged_df['Size'] != 0.0]

# Calculer la moyenne de la colonne Size
size_mean = filtered_merged_df['Size'].mean()

# Remplacer les valeurs de la colonne Size égales à 0.0 par la moyenne calculée
merged_df['Size'] = merged_df['Size'].replace(0.0, size_mean)

#======================================================================================#
#==============Je mets Free ou la valeur n'est pas égale a Free ou Paid ===============#
#======================================================================================#

# Remplacer les valeurs autres que "Free" ou "Paid" dans la colonne "Type" par "Free"
merged_df.loc[~merged_df['Type'].isin(['Free', 'Paid']), 'Type'] = 'Free'


#=================================================================================================================#
#==============Je remplace la valeur 0 par Nan et je remplace par le genre qui est le plus présent ===============#
#=================================================================================================================#

# Remplacer les valeurs 0 par NaN dans la colonne "Genre"
merged_df['Genres'] = merged_df['Genres'].replace(0, pd.NA)

# Calculer le nombre de chaque genre dans la colonne "Genre" (en excluant les valeurs NaN)
genre_counts = merged_df['Genres'].value_counts()

# Trouver le genre le plus fréquent
most_frequent_genre = genre_counts.idxmax()

# Remplacer les valeurs NaN dans la colonne "Genre" par le genre le plus fréquent
merged_df['Genres'].fillna(most_frequent_genre, inplace=True)

#============================================================================================================#
#==============J'arrondie AVG_Pol et AVG_SUB  à chiffre après la virgule'===============#
#============================================================================================================#

# Arrondir les valeurs de la colonne "AVG_Pol" à deux chiffres après la virgule
merged_df['AVG_Pol'] = merged_df['AVG_Pol'].round(2)

# Arrondir les valeurs de la colonne "AVG_Pol" à deux chiffres après la virgule
merged_df['AVG_Sub'] = merged_df['AVG_Sub'].round(2)

#========================================================#
#==============Je crée un nouveau fichier ===============#
#========================================================#

# Enregistrer les données fusionnées dans un nouveau fichier
merged_df.to_csv("merged_googleplaystore.csv", index=False)

print(merged_df.head(55))

#on verifie les types de variables
print(merged_df.dtypes)