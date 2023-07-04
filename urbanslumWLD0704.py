#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from pandas import ExcelWriter, ExcelFile
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.models import BayesianNetwork

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from tabulate import tabulate
from fpdf import FPDF

df = pd.read_csv('urbanslumWLD1.csv')
years = np.arange(2000, 2021)
proportions = np.array([31.2, np.nan, 30.9, np.nan, 30.1, np.nan, 29.2, np.nan, 28.2, np.nan, 27.3, np.nan, 26.3, np.nan, 25.4, np.nan, 24.6, np.nan, 24.4, np.nan, 24.2])

data = np.column_stack((years, proportions))
imputer = IterativeImputer()
imputed_data = imputer.fit_transform(data)

filled_years = imputed_data[:, 0]
filled_proportions = imputed_data[:, 1]

missing_values = df.isna().sum()
df = df.dropna()
df = df.fillna(method='bfill')
df['year'] = pd.to_datetime(df['year'], format='%Y').dt.year

def plot_df(df, x, y, title="", xlabel='Year', ylabel='Urban Slum Index', dpi=100):
    plt.figure(figsize=(12,6), dpi=dpi)
    plt.plot(x, y, color='tab:red', linewidth=2) 
    plt.grid(True) 
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.xticks(np.arange(min(x), max(x)+1, 1), rotation=20, fontsize=12) 
    plt.yticks(fontsize=12)
    plt.title(title, fontsize=16)  
    plt.xlabel(xlabel, fontsize=14)  
    plt.ylabel(ylabel, fontsize=14) 
    plt.savefig('Fig.3 Trend_UrbanSlumPopProp.pdf', format='pdf')
plot_df(df, x=df['year'], y=df['urbanslum'])

plt.rcParams['font.family'] = ['Arial Unicode MS']

X = df.drop("urbanslum", axis=1)
y = df['urbanslum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
train_cols = 14
train_rows = len(X_train.columns) 
plt.figure(figsize=(4*train_cols, 4*train_rows))
i = 0
failed_tests = []  
for col in X_train.columns:
    i += 1
    ax = plt.subplot(train_rows, train_cols, i)
    sns.distplot(X_train[col], fit=stats.norm)
    sns.distplot(X_test[col], fit=stats.norm)
    plt.legend(['Train', 'Test'])
    ks_statistic, p_value = stats.ks_2samp(X_train[col].dropna(), X_test[col].dropna())    
    if p_value < 0.05:
        failed_tests.append(col)
    i += 1
    ax = plt.subplot(train_rows, train_cols, i)
    res = stats.probplot(X_train[col].dropna(), plot=plt)
    stats.probplot(X_test[col].dropna(), plot=plt)   

    ax.set_title(f'KS statistic: {ks_statistic:.2f}, p-value: {p_value:.2f}')    
# plt.tight_layout()
plt.savefig('Fig4 KS test.pdf', format='pdf', bbox_inches='tight')

if 'year' in df.columns:
    df = df.drop(columns='year')
corr = df.corr()
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.clustermap(corr, cmap=cmap, center=0,
               square=True, linewidths=.5, cbar_kws={"shrink": .5, 'label': 'Correlation coefficient'})
plt.savefig('Fig.1 Clustered heatmap of the overall interrelationships among the variables.pdf', format='pdf', bbox_inches='tight') 

import pandas as pd
feature_names = ["urbanslum",
    "NV.IND.MANF.ZS",
    "SP.POP.GROW",
    "SP.URB.GROW",
    "SP.ADO.TFRT",
    "SH.STA.MALN.ZS",
    "EN.ATM.CO2E.KD.GD",
    "SH.DYN.MORT",
    "NE.CON.TOTL.KD.ZG",
    "NY.GDP.MKTP.KD.ZG",
    "BX.KLT.DINV.WD.GD.ZS",
    "FP.CPI.TOTL.ZG",
    "NV.IND.TOTL.ZS",
    "CM.MKT.LCAP.GD.ZS",
    "GC.XPN.TOTL.GD.ZS",
    "FS.AST.CGOV.GD.ZS",
    "NE.GDI.FTOT.ZS",
    "BG.GSR.NFSV.GD.ZS",
    "DT.ODA.ODAT.PC.ZS",
    "SH.IMM.MEAS",
    "SE.ENR.PRSC.FM.ZS",
    "IT.CEL.SETS.P2",
    "BM.TRF.PWKR.CD.DT",
    "SH.MMR.RISK",
    "SP.DYN.LE00.IN",
    "NY.GDP.PCAP.PP.KD",
    "NY.GNP.MKTP.PP.CD",
    "SL.GDP.PCAP.EM.KD",
    "SH.STA.SMSS.UR.ZS",
    "SP.URB.TOTL",
    "AG.SRF.TOTL.K2",
    "NE.RSB.GNFS.ZS",
    "NE.CON.GOVT.ZS",
    "NY.GNS.ICTR.ZS",
    "NY.GDP.MINR.RT.ZS"
]

file_path = "urbanslumWLD1.csv"  

try:   
    df = pd.read_csv(file_path)
    df_selected_features = df[feature_names] 
    df_selected_features.to_csv("selected_features1.csv", index=False)
    print("New CSV file 'selected_features1.csv' created successfully.")
except FileNotFoundError:
    print("Error: File 'urbanslumWLD1.csv' not found. Please provide the correct file path.")

df = df.apply(pd.to_numeric, errors='coerce')
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df)

pca = PCA()  
df_pca = pca.fit_transform(df_normalized)  

plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  
plt.title('Dataset Explained Variance')
plt.savefig('Fig5 PCA explained ratio.pdf', format='pdf', bbox_inches='tight') 

n_pcs = pca.components_.shape[0]
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = df.columns
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
scaler = StandardScaler()
df_normalized = scaler.fit_transform(df.drop('urbanslum', axis=1))
pca = PCA()  
df_pca = pca.fit_transform(df_normalized)  

n_pcs = 10 
most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
initial_feature_names = df.drop('urbanslum', axis=1).columns
most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]  
most_important_names_set = set(most_important_names)

df_pca_results = pd.DataFrame({'PC':range(1, n_pcs+1),  # Adjusted to reflect top 10
                               'Explained Variance Ratio': pca.explained_variance_ratio_[:n_pcs],  # Adjusted to get only top 10
                               'Most Important Feature': most_important_names})


df_pca_results.to_csv('Table 4. PCA and explained variance ratio.csv', index=False)


np.random.seed(42)
y = df['urbanslum']
X = df.drop('urbanslum', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', RidgeCV()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor()),
    ('Gradient Boosting', GradientBoostingRegressor())
]

results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'R2'])
for name, model in models:

    y_pred = cross_val_predict(model, X, y, cv=5)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    if r2 < 0:
        r2 = 0.0
    results.loc[len(results)] = [name, mae, mse, r2]
df = pd.read_csv('selected_features1.csv')
y = df['urbanslum']
X = df.drop('urbanslum', axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)
models = [
    ('Linear Regression', LinearRegression()),
    ('Ridge Regression', RidgeCV()),
    ('Decision Tree', DecisionTreeRegressor()),
    ('Random Forest', RandomForestRegressor()),
    ('Gradient Boosting', GradientBoostingRegressor())
]
results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'R2'])
for name, model in models:   
    y_pred = cross_val_predict(model, X, y, cv=5)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    if r2 < 0:
        r2 = 0.0   

    results.loc[len(results)] = [name, mae, mse, r2]

results.to_csv('Table5. model performace.csv', index=False)




