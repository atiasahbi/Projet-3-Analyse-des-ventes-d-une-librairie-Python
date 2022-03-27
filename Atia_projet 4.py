#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np #importation classique du numpy sous l'alias np
import pandas as pd #importation classique du pandas sous l'alias pd
import  matplotlib.pyplot as plt #importation classique du module matplotlib.pyplot sous l'alias plt
import seaborn as sns
import datetime as dt
import cufflinks as cf
import chart_studio.plotly as py
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
import  plotly.subplots
from plotly.subplots import make_subplots
from plotly.offline import iplot,plot,download_plotlyjs,init_notebook_mode
init_notebook_mode(connected=True)
cf.go_offline()
from scipy.stats import pearsonr
from importlib import reload
plt=reload(plt)
import scipy.stats
from scipy.stats import chi2_contingency
from stats import *
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind


# In[2]:


cd "C:\Users\narje\Desktop\données_p6\DAN-P6-donnees"


# # 1) Importation des exports

# In[3]:


customers = pd.read_csv('customers.csv') #importer le fichier customers
customers.head()
customers.info()
customers.describe(include = 'all')


# In[4]:


products = pd.read_csv('products.csv') #importer le fichier products
products.head()
products.info()
products.describe(include = 'all')

recherche_null = products[products ['price'].isnull()] #recherche des valeurs nulles de la variable prix

#Observation :

products [ products ['price'] < 0] #on a une ligne où le prix est négatif


# In[5]:


transactions = pd.read_csv('transactions.csv') #importer le fichier transactions
transactions.head()
transactions.info()
transactions.describe(include = 'all')

transactions [( transactions ['client_id'] == 'ct_0')] #session test (id_prod=T_0)


# # 2) Nettoyage des dataframes et vérification de l'unicité des clés

# In[6]:


#nettoyage du dataframe customers :

customers.dropna(axis = 1 , how = 'all' , inplace = True)
customers.dropna(axis = 0 ,how = 'all' , inplace = True)
customers.dropna(subset = ['client_id'] , axis = 0 ,how = 'any' , inplace = True)
print(customers.drop_duplicates(['client_id']).shape)
customers = customers.drop_duplicates(['client_id'])

print(f'client_id:{customers.client_id.nunique()}') #vérification de l'unicité du clé de la table customers

customers.info()
recherche_null = customers[ customers ['client_id'].isnull() ]
recherche_null
customers.describe(include = 'all')


# In[7]:


#nettoyage du dataframe products
products.dropna(axis = 1 , how = 'all' , inplace = True)
products.dropna(axis = 0 ,how = 'all' , inplace = True)
products.dropna(subset = ['id_prod'] , axis = 0 ,how = 'any' , inplace = True)
print(products.drop_duplicates(['id_prod']).shape)
products = products.drop_duplicates(['id_prod'])
products.describe (include = 'all')
print(f'id_prod:{products.id_prod.nunique()}') #vérification de l'unicité du clé de la table products
products.info()
recherche_null = products[ products ['id_prod'].isnull() ]
recherche_null
products.describe (include = 'all')
recherche_néga = products [( products ['price'] < 0)] #recherche des valeurs nulls du prix
recherche_néga


# In[8]:


#Nettoyage du dataframe transactions

prix_négatif = transactions ['id_prod'] == 'T_0' #recherche des valeurs abberantes de la variable price

transactions [( transactions ['client_id'] == 'ct_0')]

transactions.drop(index = transactions [ prix_négatif ].index , inplace = True) #enlever les valeurs abberantes (négatives)

recherche_null = transactions[ transactions ['id_prod'].isnull()] #recherche des identifiants produit nulls
recherche_null


#transactions.describe(include = 'all')


# # 3) La jointure des dataframes transactions et products

# In[9]:


#jointure entre la table transactions et la table products:

transactions_products = pd.merge(transactions,products, how = 'left', on = 'id_prod')

#transactions_products.describe(include='all')

transactions_products['date'] = pd.to_datetime(transactions_products['date']) #parser la colonne 'date' comme date

transactions_products.set_index ('date',inplace = True)# la colonne date comme index de transactions_products
transactions_products

#transactions_products.head()

recherche_null = transactions_products[transactions_products ['price'].isnull()]#recherche s'il y'a des valeurs nulls pour la variable price
recherche_null


# In[10]:


#Imputation par la moyenne pour la catégorie 0 du valeur manquante (prix):['0_2245']

prix_moyen_0 = transactions_products[transactions_products['categ'] == 0]['price'].mean() #calcul prix moyen pour la catég0

transactions_products['price'].fillna(prix_moyen_0, inplace=True) #imputation par la moyenne du prix pour la catég0

transactions_products[transactions_products ['id_prod']=='0_2245']

transactions_products['categ'].fillna(0.0, inplace=True) #imputation de la catég0(0.0)

transactions_products[transactions_products ['id_prod']=='0_2245']


# # 4) Analyse de CA :

# In[11]:


from importlib import reload
plt = reload(plt)

CA_total = transactions_products['price'].sum() #le CA total est la somme

print("le chiffre d'affaire total:",CA_total,"euros") #affichage de CA total

#CA par ans :

CA_2021 = transactions_products.loc['2021']['price'].sum() 
CA_2022 = transactions_products.loc['2022']['price'].sum() 
CA_2023 = transactions_products.loc['2023']['price'].sum()
print('le CA pour les années 2021,2022 et 2023 est respectivement:',CA_2021,CA_2022,CA_2023)

#CA par mois:

transactions_products.price.resample('M').sum().plot(figsize=(12,6)) #resampler le CA par mois
plt.title('CA Mensuel') #titre de la graphique
plt.xlabel('Date') #axe des abscisses
plt.ylabel('CA') #axe des ordonnées
plt.savefig('CA mensuel en fonction du tepms.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution


# # CA mensuel par categ :

# In[12]:


from importlib import reload
plt=reload(plt)
categ0 = transactions_products[transactions_products['categ'] == 0] #dataframe contenent seulement les produitd de categ0
categ1 = transactions_products[transactions_products['categ'] == 1] #dataframe contenent seulement les produitd de categ1
categ2 = transactions_products[transactions_products['categ'] == 2] #dataframe contenent seulement les produitd de categ2
plt.figure(figsize = (8,6))
plt.plot(categ0.price.resample('M').agg(['sum']),'b.-',label = 'categ 0') #grahe CA par mois categ0
plt.plot(categ1.price.resample('M').agg(['sum']),'y.-',label = 'categ 1') #grahe CA par mois categ1
plt.plot(categ2.price.resample('M').agg(['sum']),'g.-',label = 'categ 2') #grahe CA par mois categ2
plt.title('CA mensuel par catégorie')#titre de la figure
fontweight='bold'
fontsize=22
plt.xlabel('Date') #axe des abscisses
plt.ylabel('CA') #axe des ordonnées
plt.legend()
plt.savefig('CA mensuel par catégorie.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# # Graphe CA par categ et par semaine :

# In[13]:



plt=reload(plt)
plt.figure(figsize = (12,6))

plt.plot(categ0.price.resample('W').agg(['sum']),label = 'Evolution de CA hebdomadaire pour la categ 0') #resampler le CA par seamine pour la categ 0
plt.plot(categ1.price.resample('W').agg(['sum']),label = 'Evolution de CA hebdomadaire pour la categ 1') #resampler le CA par seamine pour la categ 1
plt.plot(categ2.price.resample('W').agg(['sum']),label = 'Evolution de CA hebdomadaire pour la categ 2') #resampler le CA par seamine pour la categ 2

plt.title('CA par categ et par semaine') #titre de la figure
plt.xlabel('Date') #axe des abscisses
plt.ylabel('CA') #axe des ordonnées
plt.savefig('CA par categ et par semaine.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.legend()
plt.show()


# # Graphe CA par categ et par jour :

# In[14]:


plt=reload(plt)
plt.figure(figsize = (20,6))

plt.plot(categ0.price.resample('D').agg(['sum']),label = 'Evolution de CA journalier pour la categ 0') #resampler le CA par jour
plt.plot(categ1.price.resample('D').agg(['sum']),label = 'Evolution de CA journalier pour la categ 1')
plt.plot(categ2.price.resample('D').agg(['sum']),label = 'Evolution de CA journalier pour la categ 2')

plt.title('CA par categ et par jour') #titre de la figure
plt.xlabel('Date') #axe des abscisses
plt.ylabel('CA') #axe des ordonnées
plt.savefig('CA par categ et par jour.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.legend()
plt.show()


# # Les valeurs abberantes de CA par la méthode de Z score:

# In[15]:


CA_journalier = transactions_products.loc[:,'price'].resample('D').agg(['sum']) #calcul du CA journalier 

CA_journalier.rename (columns = {"sum":"CA journalier"} , inplace = True)

CA_journalier['zscore'] = ((CA_journalier['CA journalier'] - CA_journalier['CA journalier'].mean()) 
                           / CA_journalier['CA journalier'].std()).abs() #calcul du Z score

valeurs_aberrantes = CA_journalier [ CA_journalier ["zscore"] >2 ] #le nombre des outliers du CA

valeurs_aberrantes['CA journalier'].hist(bins = 30) #graphique qui présente les valeurs abberantes avec 30 bars
plt.title('Distribution des outliers du CA') #titre de la figure
plt.xlabel("CA") #axe des abscisses
plt.ylabel('Nombre') #axe des ordonnées
plt.savefig('variation du CA en fonction du Nombre.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de la figure dans le répertoire de travail et augmentation de la résolution
print("Le nombre des valeurs abérrantes de CA journalier est de",valeurs_aberrantes.shape)


# In[16]:


#on se crée 2 dataframe pour séparer les couleurs :  

valeurs_normales = CA_journalier[(CA_journalier['zscore'] < 2 )]
valeurs_abberantes = CA_journalier[(CA_journalier['zscore'] > 2 )]

#graphique :

plt.scatter (valeurs_normales['CA journalier'].index , valeurs_normales['CA journalier'].values)
plt.scatter (valeurs_abberantes['CA journalier'].index , valeurs_abberantes['CA journalier'].values)
plt.title('Les valeurs abberantes du CA en bleu') #titre de la figure
plt.xlabel("Date") #axe des abscisses
plt.ylabel('CA') #axe des ordonnées
plt.savefig('Les valeurs abberantes du CA.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution


# In[51]:


CA_journalier.boxplot(column = ['CA journalier'] , grid = False)#analyse CA par client

plt.title('Boîte à moustache pour la variable CA') #titre de la figure
plt.savefig('Boîte à moustache pour la variable CA.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution


# In[18]:


#Moyenne mobile:

CA_journalier = transactions_products.loc[:,'price'].resample('D').agg(['sum']) #calcul du CA journalier 
CA_journalier.rename (columns = {"sum":"CA journalier"} , inplace = True)

moyenne_mobile = CA_journalier.loc[:,'CA journalier'].rolling(window = 10,center = True).mean()
moyenne_mobile = pd.DataFrame(data = moyenne_mobile) #convertir moyenne_mobile en dataframe

#Graphique :

plt.figure(figsize = (15,6))
plt.plot (CA_journalier["CA journalier"].index , CA_journalier["CA journalier"].values ,label = 'Evolution de CA')
plt.plot (moyenne_mobile["CA journalier"].index , moyenne_mobile["CA journalier"].values,label = 'Moyenne mobile',lw = 2)
plt.title('Evolution CA et MB') #titre de la figure
plt.xlabel('Date') #axe des abscisses
plt.ylabel('CA') #axe des ordonnées
plt.savefig('Evolution CA et MB.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.legend()
plt.show()


# # Les tops et les flops des références:

# In[19]:


plt=reload(plt)

A = transactions_products.groupby(['id_prod'])['price'].sum() 

#transactions_products.describe(include='all')

A = pd.DataFrame(data = A) #convertir A en dataframe
A.rename(columns={'price':'CA réalisé'},inplace=True)
A = A.sort_values(ascending = False , by = ['CA réalisé']) #tri

tops = A [ A ['CA réalisé'] >= 50000] #on a consideré comme tops les réferences qui ont réalisées un CA supérieur à 50000 euros

tops.plot(kind='bar',title='Les tops des références',color='b')
plt.title('Les tops des références',fontweight='bold',fontsize=22)
plt.savefig("Les tops des références.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()

flops = A [ A ['CA réalisé'] < 5] #on a consideré comme flops les réferences qui ont réalisées un CA inférieur à 5 euros

flops.plot(kind='bar',title='Les flops des références',color='b')
plt.title('Les flops des références',fontweight='bold',fontsize=22)
plt.savefig("Les flops des références.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# # Répartition du chiffre d'affaires entre les clients:

# In[20]:


#Jointure entre customers et transactions_products :

plt.style.use('ggplot')
customers.describe(include = 'all')
transactions_products.describe(include = 'all')

DF_total = pd.merge(transactions_products,customers,how = 'left' , on = 'client_id') #jointure transactions_products et customers

#Courbe de lorenz :

C =  DF_total.groupby(['client_id'])['price'].agg(['sum'])
C = C.sort_values(ascending = True , by = ['sum'])
ventes = C['sum'].values
n = len(ventes)
lorenz = np.cumsum(np.sort(ventes)) / ventes.sum()
lorenz = np.append([0],lorenz) # La courbe de Lorenz commence à 0
plt.axes().axis('equal')
xaxis = np.linspace(0-1/n,1+1/n,n+1) #Il y a un segment de taille n pour chaque individu, plus 1 segment supplémentaire d'ordonnée 0. Le premier segment commence à 0-1/n, et le dernier termine à 1+1/n.
x = np.linspace(0, 1)

plt.plot(xaxis , lorenz,drawstyle = 'steps-post', label = 'Courbe de Lorenz')
plt.plot(x , x,label = 'Première bissectrice') # courbe première bissectrice
plt.xlabel('Part cumulé en pourcentage des clients') #axe des abscisses
plt.ylabel('Part cumulé en pourcentage du CA') #axe des ordonnées
plt.savefig('courbe de lorenz.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.title('Répartition du CA par client')
plt.legend()
plt.show()


# # Les valeurs abberantes de la variable CA par client :

# In[21]:


C =  DF_total.groupby(['client_id'])['price'].agg(['sum']) #Calcul chiffre d'affaire par client

C.boxplot(column = 'sum',figsize = (5,4)) #Graphe illustre le CA réalisé par client

plt.title('Les valeurs abberantes de la variable CA par client') #titre de la figure
plt.savefig('Les valeurs abberantes de la variable CA par client.png' , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution


# # La contribution des 4 grands clients au CA total:

# In[22]:



D = transactions_products.groupby(['client_id'])['price'].agg(['sum'])#CA réalisé par chaque client

clients_pro = D [ D ['sum'] >= 100000] #les 'grands clients'
E = clients_pro['sum'].sum() #CA réalisé par les 4 grands clients
print('La contribution des 4 grands clients au CA total est de',(E/CA_total)*100,'%')

labels = ['Gros clients','CA total']
plt.pie([E,CA_total-E],labels = labels,autopct='%2.2f%%') #La contribution de 4 grands clients à la CA total
plt.title('Contribution des 4 grands clients au CA total',fontweight='bold',fontsize=15) #titre du graph
plt.savefig("Contribution des 4 grands clients au CA total.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()
clients_pro


# # Eliminer le 4 grands clients de la base :

# In[23]:



Masque = DF_total[(DF_total.client_id == 'c_1609') |
(DF_total.client_id =='c_4958') |
(DF_total.client_id =='c_6714') |
(DF_total.client_id =='c_3454')] #masque pour selectionner les 4 grands clients

DF_finale = DF_total.drop(Masque.index) #supression de 4 grands clients (Valeurs abberantes)
DF_finale[DF_finale['client_id']=='c_3454']


# In[24]:


D =  DF_finale.groupby(['client_id'])['price'].agg(['sum']) #CA réalisé par chaque client

D.boxplot(column = ['sum'],figsize = (5,3))
plt.title('Les valeurs abberantes de la variable CA par client aprés suppression de 4 clients') #titre de la figure
plt.savefig('Les valeurs abberantes de la variable CA par client aprés suppression de 4 clients.png' , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution


# # Distribution des ages des clients :

# In[25]:


plt=reload(plt)
L = DF_finale.groupby(['birth'])['birth'].agg(['count']) # Nombre de clients pour chaque age

DF_finale['age'] = 2021 - DF_finale.birth #Ajout d'une colonnes Age

DF_finale.age.hist( bins = 30 , color = 'b') #Graphe de distribution des ages des clients
plt.xlabel('Age') #axe des abscisses
plt.ylabel('Nombre') #axe des ordonnées
plt.title("Distribution des ages des clients") #titre de la figure
plt.savefig("Distribution des ages des clients.png" , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution


# # Panier moyen par catégorie :

# In[26]:


plt=reload(plt)
panier_moyen = DF_finale.groupby(['session_id','categ'])['price'].agg(['sum']) #calcul du panier moyen

panier_moyen.reset_index(inplace = True) #reindexation de dataframe M

sns.boxplot(data = panier_moyen ,x = 'categ',y = 'sum') #graphe qui ulistre le panier moyen par catégories
plt.xlabel('Catégorie') #axe des abscisses
plt.ylabel('Panier moyen') #axe des ordonnées
plt.title('Montant du panier moyen par catégorie')
plt.savefig("Montant du panier moyen par catégorie.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[27]:


#Analyse de l'indépendance entre les panier moyens et les catégorie des livres : ANOVA


#Hypothèse0 : le 2 variable sont indépendant
#Hypothèse1 : dépendance entre les 2 variables

# On admet que si p<0.05 donc on rejette la H0 et on dit que notre statistique n'est pas du au hasard*


X = "categ" # qualitative
Y = "sum" # quantitative

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
Eta = eta_squared(panier_moyen[X],panier_moyen[Y]) #à priopri on a une corrélation entre les 2 variables et la catég 2 a des prix plus élévé % aux autres
print(Eta)

#calcul p value
scipy.stats.f_oneway(panier_moyen['sum'][panier_moyen['categ'] == 0.0],
               panier_moyen['sum'][panier_moyen['categ'] == 1.0],
               panier_moyen['sum'][panier_moyen['categ'] == 2.0])

#p<0.05 donc on rejette H0 et on admet que les 2 variables sont corrélées


# In[28]:


DF_finale.corr()


# # Répartition du CA par ages des clients :

# In[29]:


plt.style.use('ggplot')
F = DF_finale.groupby(['birth'])['price'].agg(['sum','mean','count']) #dataframe qui regroupe le CA par age client

F['age'] = 2021 - F.index #ajout d'une colonne age 

plt.scatter (F.age , F['sum'], c = 'blue', alpha=1) #graphe nuages des points age/CA
plt.xlabel('Age') #axe des abscisses
plt.ylabel('CA') #axe des ordonnées #/len()#????
plt.title("Répartition du CA par ages des clients") #titre de la figure
plt.savefig("Répartition du CA par ages des clients.png" , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution


# In[30]:


#Analyse de la corrélation entre l'age des clients et le CA réalisé :

#Hypothèse0 : le CA réalisé par un client  n'est pas corrélé  au son age 
#Hypothèse1 : les 2 variables sont corrélées 

r,p = pearsonr(F["age"],F["sum"])
print('Le coefficient de pearson et p value :',pearsonr(F["age"],F["sum"]))
print('La covariance est de:',np.cov(F["age"],F["sum"])[1,0])

#p<0.05 donc on rejette H0 et on admet que les 2 variables sont corrélées


# # Répartition CA par genre: 

# In[31]:


plt.style.use('ggplot')

M = DF_finale.groupby(['sex'])['price'].agg(['sum']) #dataframe qui regroupe le CA par genre des clients

M.reset_index(inplace = True) #reindexation de dataframe M

Hommes = M.loc[M.sex == 'm'].sum()[1] #masque sur les masculins

Femmes = M.loc[M.sex == 'f'].sum()[1] #masque sur les féminins
labels = ['Hommes','Femmes']
plt.pie([Hommes,Femmes],labels = labels,autopct='%2.3f%%') #graphe en camembert contient la participation en CA par genre
plt.title('CA par genre',fontweight='bold',fontsize=22) #titre du graph
plt.savefig("CA par genre.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[32]:


#Répartition genre(de point de vue nombre) / client

m = DF_finale.groupby(['sex'])['client_id'].agg(['nunique'])

m.reset_index(inplace = True) #reindexation de dataframe m

Hommes = m.loc[m.sex == 'm'].sum()[1] #masque sur les masculins
Femmes = m.loc[m.sex == 'f'].sum()[1] #masque sur les féminins

labels = ['masculins','féminins']

plt.pie([Hommes,Femmes],labels = labels,autopct='%2.3f%%') #graphe en camembert contient la participation en CA par genre
plt.title('Nombre des clients par genre',fontweight='bold',fontsize=22) #titre du graph
plt.savefig("Nombre des clients par genre.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[33]:


#Analyse de l'indépendance entre le CA et le genre : ANOVA


#Hypothèse0 : le 2 variable sont indépendant
#Hypothèse1 : dépendance entre les 2 variables

# On admet que si p<0.05 donc on rejette la H0 et on dit que notre statistique n'est pas du au hasard*


X = "categ" # qualitative
Y = "price" # quantitative

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
Eta = eta_squared(DF_finale[X],DF_finale[Y]) #à priopri on a une corrélation entre les 2 variables
print(Eta)

#calcul p value
scipy.stats.f_oneway(DF_finale['price'][DF_finale['sex'] == 'm'],
               DF_finale['price'][DF_finale['sex'] == 'f'])
           

#p>0.05 donc on admet H0 et on dit que les 2 variables ne sont pas corrélées


# # Comparaison des paniers moyens par client:

# In[34]:


E = DF_finale.groupby(['sex','session_id'])['price'].agg(['sum']) 
N = E.groupby(['sex'])['sum'].agg(['mean']) #calcul panier moyen par client
N.reset_index(inplace = True) #reindexation de dataframe N
E.reset_index(inplace = True) #reindexation de dataframe E

sns.boxplot(data = E,x = 'sex',y = 'sum')#graphe panier moyen/client

plt.xlabel('Catégorie') #axe des abscisses
plt.ylabel('Panier moyen') #axe des ordonnées
plt.title('Comparaison des paniers moyens par client')#titre
plt.savefig("Comparaison des paniers moyens par client.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[35]:


#Analyse de l'indépendance entre les panier moyens et les catégorie des livres : ANOVA


#Hypothèse0 : le 2 variable sont indépendant
#Hypothèse1 : dépendance entre les 2 variables

# On admet que si p<0.05 donc on rejette la H0 et on dit que notre statistique n'est pas du au hasard*


X = "sex" # qualitative
Y = "sum" # quantitative

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
Eta = eta_squared(E[X],E[Y]) #à priopri on a pas une corrélation entre les 2 variables eta trés faible
print(Eta)

#calcul p value
scipy.stats.f_oneway(E['sum'][E['sex'] == 'm'],
               E['sum'][E['sex'] == 'f'])
              

#p<0.05 et eta_squared trés faible donc y'a pas de relation de dependance entre les 2 variables


# # Répartition par catégorie pour chaque genre :

# In[36]:



Femmes = DF_finale[DF_finale['sex'] == 'f'].groupby(['categ'])['price'].agg(['sum']) #masque pour selectionner les femmes
Femmes.reset_index(inplace = True) #reindexation de dataframe Femmes

Hommes = DF_finale[DF_finale['sex'] == 'm'].groupby(['categ'])['price'].agg(['sum']) #masque pour selectionner les hommes
Hommes.reset_index(inplace = True) #reindexation de dataframe Femmes

fig = make_subplots(rows = 1, cols = 2, specs=[[{"type": "pie"}, {"type": "pie"}]]) #figure illustre catég/genre
fig.add_trace(go.Pie(labels = Femmes['categ'], values=Femmes['sum'], title = 'Femmes'),1,1)
fig.add_trace(go.Pie(labels = Hommes['categ'], values=Hommes['sum'], title = 'Hommes'),1, 2)
fig.update_layout(title = "Répartition par catégorie pour chaque genre", title_x = 0.5) #titre du figure              
fig.show()


# In[37]:


#Calul de chi2 et p value :
X = "sex"
Y = "categ"
cont = DF_finale[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")

#H0:il y'a pas de dépendance entre le genre et la catégorie acheté
#Ha:il y'a une corrélation entre les 2 variables

scipy.stats.chi2_contingency(cont)

#chi2=20.20 et p=0.0025 donc on rejette H0 et on admet qu'il y'a une dépendance entre les 2 variables


# In[38]:



#Carte de chaleur :

X = "sex"
Y = "categ"
cont = DF_finale[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,margins=True,margins_name="Total")

tx = cont.loc[:,["Total"]]
ty = cont.loc[["Total"],:]
n = len(DF_finale)
indep = tx.dot(ty) / n

c = cont.fillna(0) # On remplace les valeurs nulles par 0
measure = (c-indep)**2/indep
xi_n = measure.sum().sum()
table = measure/xi_n
sns.heatmap(table.iloc[:-1,:-1],annot=c.iloc[:-1,:-1])
plt.show()
indep


# # Dispersion de fréquence d'achat par sexe :

# In[39]:


s=DF_finale.groupby(['sex','client_id'])['session_id'].agg(['nunique'])


s.reset_index(inplace = True) #reindexation de dataframe C
s[s['client_id']=='c_1000']
s
sns.boxplot(data = s,x = 'sex',y = 'nunique') #graphe fréquence d'achat/genre
plt.title("Dispersion de fréquence d'achat par sexe")
plt.xlabel('Genre')
plt.ylabel("Nombre d'achat")
plt.savefig("Dispersion de nombre d'achat par genre.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[40]:


#Analyse de l'indépendance entre les fréquences d'achat et le genre : ANOVA


#Hypothèse0 : le 2 variable sont indépendant
#Hypothèse1 : dépendance entre les 2 variables

# On admet que si p<0.05 donc on rejette la H0 et on dit que notre statistique n'est pas du au hasard*

X = "sex" # qualitative
Y = "nunique" # quantitative

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
Eta = eta_squared(s[X],s[Y]) #à priopri on a une corrélation entre les 2 variables
print(Eta)

#calcul p value
scipy.stats.f_oneway(s['nunique'][s['sex'] == 'm'],
               s['nunique'][s['sex'] == 'f'])
              

#p>0.05 donc on admet H0 et on dit que les 2 variables ne sont pas corrélées


# # Répartition du panier moyen selon l'age :

# In[41]:


B = DF_finale.groupby(['age'])['price'].agg(['sum','mean','count']) #calcul panier moyen

B.reset_index(inplace = True) #reindexation de dataframe B

plt.scatter (B.age , B['mean'], c = 'blue', alpha = 1 ) 
plt.style.use('ggplot')
plt.title("Répartition du panier moyen selon l'age") #titre de la figure
plt.xlabel("Age") #axe des abscisses
plt.ylabel('panier moyen') #axe des ordonnées
plt.savefig("Répartition du panier moyen selon l'age.png" , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[42]:


#Analyse de la corrélation entre l'age des clients et le panier moyen :


#Hypothèse0 : le panier moyen réalisé par un client  n'est pas corrélé  au son age 
#Hypothèse1 : les 2 variables sont corrélées 

print('Le coefficient de pearson et le p value :',pearsonr(B["age"].values,B["sum"].values))
print('La covariance est de:',np.cov(B["age"],B["sum"])[1,0])

#p<0.05 donc on rejete H0 et on admet H1 

#coorélation négative fortement significative


# # Dispersion de la fréquence d'achat selon l'age :

# In[43]:


L = DF_finale.groupby(['age','client_id'])['session_id'].agg(['nunique']) #
L.reset_index(inplace = True) #reindexation de dataframe L

L.loc[(L['age'] >=0) & (L['age'] <=29), "tranche d'age"] = '<29'   #nouvelle colonne qui contient les tranches d'ages
L.loc[(L['age'] >29) & (L['age'] <=50), "tranche d'age"] = '30-50'
L.loc[L['age']>50, "tranche d'age"] = '>50'

sns.boxplot(data = L , x = "tranche d'age" , y = 'nunique') #graphe fréquence d'achat / tranches d'ages
plt.xlabel("Tranches d'ages") #axe des abscisses
plt.ylabel("Nombre d'achat") #axe des ordonnées
plt.title("Dispersion de nombre d'achat selon l'age")
plt.savefig("Dispersion de nombre d'achat selon les ages.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()
L


# In[44]:


#Analyse de l'indépendance entre les fréquences d'achat et l'age  : ANOVA


#Hypothèse0 : le 2 variable sont indépendant
#Hypothèse1 : dépendance entre les 2 variables

# On admet que si p<0.05 donc on rejette la H0 et on dit que notre statistique n'est pas du au hasard*

X = "tranche d'age" # qualitative
Y = "nunique" # quantitative

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
Eta = eta_squared(L[X],L[Y]) #à priopri on a une corrélation entre les 2 variables
print(Eta)

#calcul p value
scipy.stats.f_oneway(L['nunique'][L["tranche d'age"] == '<29'],
               L['nunique'][L["tranche d'age"] == '30-50'],L['nunique'][L["tranche d'age"] == '>50'])
              

#p<0.05 donc on rejette H0 et on admet que les 2 variables sont corrélées


# # Dispersion des ages par catégorie :

# In[50]:


sns.boxplot(data=DF_finale,x = 'categ',y = 'age') #graphe age/catégorie

plt.title('Dispersion des ages par catégorie') #titre de la figure
plt.xlabel("Catégorie") #axe des abscisses
plt.ylabel("Age") #axe des ordonnées
plt.savefig("Dispersion des ages des clients par catégorie.png" , dpi = 200,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[46]:


#Analyse de la corrélation Age par catégorie : ANOVA


#Hypothèse0 : le 2 variable sont indépendant
#Hypothèse1 : dépendance entre les 2 variables

# On admet que si p<0.05 donc on rejette la H0 et on dit que notre statistique n'est pas du au hasard*

X = "categ" # qualitative
Y = "age" # quantitative

def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT
    
Eta = eta_squared(DF_finale[X],DF_finale[Y]) #à priopri on a une corrélation entre les 2 variables
print(Eta)

#calcul p value
scipy.stats.f_oneway(DF_finale['age'][DF_finale["categ"] == 0.0],
               DF_finale['age'][DF_finale["categ"] == 1.0],DF_finale['age'][DF_finale["categ"] == 2.0])
              

#p<0.05 donc on rejette H0 et on admet que les 2 variables sont corrélées


# # Probabilité :

# In[47]:


p1=DF_finale[DF_finale['id_prod']=='0_525']['client_id']
p2=DF_finale[DF_finale['id_prod']=='2_159']['client_id']

p1 = pd.DataFrame(data = p1)
p2 = pd.DataFrame(data = p2)

prob = pd.merge(p2,p1, how = 'inner', on = 'client_id') #jointure interne entre p1 et p2
# P(2_159 & 0_525) == P(0_525). Donc la probabilité qu'on cherche est donnée par le rapport P(0_525) / P(2_159).

proba = ((p1.nunique()/p2.nunique()) *100).round(2)
proba = pd.DataFrame(data = proba) #convertir proba en dataframe

print('La probabilité qu’un client achète la référence 0_525 sachant qu’il a acheté la référence 2_159 =',proba.loc['client_id',0],'%')

