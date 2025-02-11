# Exercice d'évaluation pour le poste Inria / AP-HP

L'objectif de ce travail est d'évaluer vos compétences sur une
mise en situation proche des problématiques de la collaboration
entre Inria et AP-HP.

Vous avez accés à un fichier 'data.db' qui est une base de données
sqlite contenant 2 tables. Une table de patients et une table
de tests PCR (test utlisé pour le diagnostic du Covid19).

Voici un bout de code Python servant à lire les tables avec
Pandas.

```python

import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///data.db', echo=False)
con = engine.connect()
df_patient = pd.read_sql('select * from patient', con=con)
df_pcr = pd.read_sql('select * from test', con=con)
con.close()
```

L'objectif général de cette mise en situation est de comprendre les
problèmes de qualités de données. Vous devrez mettre en évidence les
doublons dans la base patients et estimez le nombre de patients positifs
au Covid19 (par tranche d'age, par localisation géographique etc.).

Les données sont synthétiques et correspondent à la géographie
de l'Australie.

## Questions

- Ecrire un notebook jupyter qui met en évidence les problèmes
de qualité de données (données incohérentes, données manquantes etc.)

- Ecrire une fonction Python `detect_duplicates` qui prend
en parametère le dataframe `df_patient` et qui renvoit
un nouveau dataframe après suppression des doublons. Vous
estimerez le pourcentage de données dupliquées. Attention,
les données dupliquées ne sont pas identiques. Il faut imaginer
des problèmes de saisies de données (typos, information manquante
etc.)

- Ecrire une ou plusieurs fonctions de test (eg utilisant https://docs.pytest.org/en/stable/)
afin de tester la qualité de votre function.

- Faire un notebook jupyter d'analyse exploratoire de données
(EDA exploratory data analysis). Il est demandé de représenter
visuellement (graphiques, histogrammes, etc.). Ce notebook
devra utiliser le dataframe `df_patient` après la déduplication
utilisant la fonction `detect_duplicates`. Vous ferez
la jointure entre les dataframes `df_patient` et `df_pcr`
afin de répresenter et discuter la prévalence de la maladie
dans la population.

## Instructions

- Vous avez le droit d'utiliser toutes les librairies présentes
dans l'écosystème Python/PyData. Ces librairies sont typiquement
disponibles sur PyPi ou conda forge.

- Le rendu de votre travail consistera en un repository sur GitHub.

- Le repository GitHub devra contenir un fichier readme.md qui
détaille le contenu des différents fichiers et les librairies
à installer pour utiliser votre code.

- Il est préférable que le code Python soit écrit en Anglais.
Les commentaires et l'analyse des résultats dans les notebooks
devront être en Français.


## Travail soumis
### Dossier de travail
Le dossier de travail contient les scripts : 
* pre_processing.ipynb : Un premier notebook correspondant au nettoyage des 
données.
* auxiliary_functions.py : Un ensemble de fonctions auxiliaires pour la
de-dpulication des données. Ce script contient aussi les tests unitaires des 
fonctions définies. 
* data_viz.ipynb : Un notebook pour la visualisation des données propres. 

Le dossier de travail contient aussi un ensemble de données
dans le dossier Data:  
* data.db : la base de données brutes (dans laquelle on a rajouté un table appelée 
clean_data correspondant aux données nettoyées).
* au_postcode.csv : une table faisant le lien entre les postcode 
et l'état (servant pour la complétion des données). 
* state_info.csv : une table contenant des informations sur la population de 
chaque état australien. 
* dup_candidates.pkl : un fichier pickle contenant un dataframe avec les 
informations sur les duplicats des données brutes. 
* state_shapefiles : un ensemble de fichiers géographiques servant à afficher
les données sur une carte par état. 
* postcode_shapefiles : un ensemble de fichiers géographiques servant à afficher
les données sur une carte par postcode.

### Dépendances
* python 3.8.3
* fuzzywuzzy 0.18.0
* pandas 1.0.5
* numpy 1.18.5
* geopandas 0.6.1
* sqlalchemy 1.3.18
* statsmodels 0.11.1
* seaborn 0.10.1
* matplotlib 3.2.2
* datetime
* multiprocessing
* unittest
* re
* time

