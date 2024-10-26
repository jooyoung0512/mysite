#%config InlineBackend.figure_format = 'svg'
#%matplotlib inline

'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
from sklearn.preprocessing import Normalizer, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import tree
from tabulate import tabulate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from pandas.plotting import scatter_matrix

salary_table = pd.read_csv("NBA_season1718_salary.csv",encoding = 'utf-8')
seasons = pd.read_csv("Seasons_Stats.csv.zip",encoding = 'utf-8')
salary_table = salary_table[['Player','season17_18']]
salary_table.rename(columns={'season17_18':'salary17_18'},inplace = True) #variable rename
salary_table['salary17_18'] = salary_table['salary17_18']/1000000 #transform salary to 'million'

seasons = seasons[seasons['Year']>=2017]
stats17 = seasons[['Year','Player','Pos','Age','G','PER',
                   'MP','PTS','AST','TRB','TOV','BLK','STL']]

stats17.drop_duplicates(subset=['Player'], keep='first',inplace=True) #drop duplicate data

c = ['MPG','PPG','APG','RPG','TOPG','BPG','SPG']
w = ['MP','PTS','AST','TRB','TOV','BLK','STL']

for i,s in zip(c,w):
    stats17[i] = stats17[s] / stats17['G']

stats17.drop(w,axis=1,inplace=True)
#stats17.drop(['G'],axis=1,inplace=True)
stats17.loc[stats17['Pos'] == 'PF-C','Pos'] = 'PF'
stats_salary = pd.merge(stats17, salary_table)
stats_salary.columns
stats_salary.drop_duplicates(subset=['Player'],keep='first',inplace=True)
stats_salary.sort_values(by='PPG',ascending=False,inplace = True)
stats_salary[['Player','PPG']].head(10)
stats_salary.sort_values(by='PER',ascending = False,inplace = True)
stats_salary[['Player','PER']].head(10)
stats_salary.sort_values(by='Age',ascending = False,inplace = True)
stats_salary[['Player','Age']].head(10)
stats_salary.sort_values(by='TOPG',ascending=False,inplace = True)
stats_salary[['Player','TOPG']].head(10)
sns.set_style("white")
heat_salary= stats_salary[['salary17_18','Pos','MPG','PPG','APG','RPG','TOPG',
                           'BPG','SPG','Age','PER']]
numeric_columns = heat_salary.select_dtypes(include=['float64', 'int64'])
dfData = numeric_columns.corr() 
sns.heatmap(dfData)

sns.lmplot(x="Age", y="PPG",hue="Pos",col="Pos",col_wrap=3,
           data=stats_salary,lowess=True).set(
    xlabel='Position',
    ylabel='Average Points Per Game')
sns.boxplot(x="Pos", y="TOPG", data=stats_salary).set(
    xlabel='Position',
    ylabel='Average Turnovers Per Game')
sns.violinplot(x="Pos", y="BPG", data=stats_salary).set(
    xlabel='Position',
    ylabel='Average Blocks Per Game')
#from mpl_toolkits.mplot3d import Axes3D
#sns.pairplot(heat_salary)
#scatter_matrix(heat_salary)
scatter_matrix(heat_salary, alpha=0.2, figsize=(10,10), diagonal='kde')
salary_table['salary17_18'].describe()
plt.hist(stats_salary['salary17_18'],density=True,bins=50)
plt.xlabel('2017-2018 Salary(million)')
plt.ylabel('Density')
plt.show()

def convert_dummy(df, feature,rank=0): 
    pos = pd.get_dummies(df[feature], prefix=feature)
    mode = df[feature].value_counts().index[rank]
    biggest = feature + '_' + str(mode)
    pos.drop([biggest],axis=1,inplace=True)
    df.drop([feature],axis=1,inplace=True)
    df=df.join(pos)
    return df


stats_salary = convert_dummy(stats_salary,'Pos')



#from sklearn.externals.six import StringIO
#import pydotplus
#import graphviz
#from IPython.display import Image
#import os

stats_salary = stats_salary.dropna()
Y = stats_salary['salary17_18']
X = stats_salary.drop(['salary17_18','Year', 'Player'],axis=1)
X.columns
# Now let us rescale our data
transformer = MaxAbsScaler().fit(X) # Scale each feature by its maximum absolute value.
newX = transformer.transform(X)
newX = pd.DataFrame(newX,columns = X.columns)

# transformed data

#transformer = MinMaxScaler().fit(heat)
#newX = transformer.transform(heat)
#newX = pd.DataFrame(newX)
#scatter_matrix(newX, alpha=0.2, figsize=(10,10), diagonal='kde')
#newX.head()
#X.head()

#transformer = RobustScaler().fit(heat)
#newX = transformer.transform(heat)
#newX = pd.DataFrame(newX)
#scatter_matrix(newX, alpha=0.2, figsize=(10,10), diagonal='kde')
#sns.pairplot(pd.DataFrame(newX))
#newX

#from sklearn.preprocessing import StandardScaler
#transformer = StandardScaler().fit(heat)
#newX2 = transformer.transform(heat)
#newX2 = pd.DataFrame(newX2)
#scatter_matrix(newX2, alpha=0.2, figsize=(10,10), diagonal='kde')
#newX2

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3)
x_train_NEW, x_test_NEW, y_train_NEW, y_test_NEW = train_test_split(newX, Y, test_size = 0.3)
# Define a function to evaluation regression algorithms, model is fitted algorithms
# predict is for if display comparison of prediction and true value of test data.
def RegEvaluation(model, ytest, xtest, nameindex, yname,totaldt, predict=True):
    ypred = model.predict(xtest)
    xtest['Pred_Y'] = model.predict(xtest)
    dt = pd.merge(totaldt,xtest,how = 'right')
    xtest.drop(['Pred_Y'],axis=1,inplace=True)
    dt = dt[[nameindex, yname,'Pred_Y']]
    dt.sort_values(by = yname, ascending = False,inplace=True)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    r2 = r2_score(ytest, ypred)
    print('RMSE is', rmse)
    print('R sequared is', r2)
    if predict:
        return dt.head(20)
clf = tree.DecisionTreeRegressor(max_depth=4, criterion="mse")
dtree = clf.fit(x_train, y_train)

RegEvaluation(dtree, y_test, x_test, 'Player', 'salary17_18',stats_salary)
dtree = clf.fit(x_train_NEW, y_train_NEW)
RegEvaluation(dtree, y_test_NEW, x_test_NEW,
              'Player', 'salary17_18',stats_salary,predict=False)
sns.set_style("whitegrid")
values = sorted(zip(x_train.columns, clf.feature_importances_), key=lambda x: x[1] * -1)
imp = pd.DataFrame(values,columns = ["Name", "Score"])
imp.sort_values(by = 'Score',inplace = True)
sns.scatterplot(x='Score',y='Name',linewidth=0,
                data=imp,s = 30, color='red').set(
    xlabel='Importance',
    ylabel='Variables')
stats_salary = stats_salary.dropna()

reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5),
                          n_estimators=500)
adaboost = reg.fit(x_train, y_train)

RegEvaluation(adaboost, y_test, x_test, 'Player', 'salary17_18',stats_salary)
ada = reg.fit(x_train_NEW, y_train_NEW)
RegEvaluation(ada, y_test_NEW, x_test_NEW,
              'Player', 'salary17_18',stats_salary,predict=False)
values = sorted(zip(x_train.columns, reg.feature_importances_), key = lambda x: x[1] * -1)
imp = pd.DataFrame(values,columns = ["Name", "Score"])
imp.sort_values(by = 'Score',inplace = True)
sns.scatterplot(x='Score',y='Name',linewidth=0,
                data=imp,s = 30, color='red').set(
    xlabel='Importance',
    ylabel='Variables')
#?pd.DataFrame.sort_values
'''

def get_player():
    return {
        "name": "ㄴㅇㄹㄴㅇㄹ"
    }
    print(f"선수이름: {name}")
    print(f"나이: {age}")
    print(f"팀: {team}")
    print(f"포지션: {position}")

# 예시 선수 정보
player_name = "홍길동"
player_age = 25
player_team = "서울 FC"
player_position = "미드필더"
