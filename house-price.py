#%% Import library
#-----Importing Library-----#
INPUT_PATH = "Datasets/"
MODEL_PATH = "Model/"

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, norm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import time
import pickle
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


def save_model(model, filename):
    filename = MODEL_PATH + filename + '.pkl'
    pickle.dump(model, open(filename, 'wb'))

def load_model(filename):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

#%% Get all dataset from files
#-----Getting Dataset-----#
train = pd.read_csv(INPUT_PATH + 'AmesHousing.csv')
# test = pd.read_csv(INPUT_PATH + 'test.csv')
train.drop(columns = ['Order'], axis=1, inplace=True)
train.drop(columns = ['PID'], axis=1, inplace=True)
train.columns = train.columns.str.replace(' ', '')

quantitive = [f for f in train.columns if train.dtypes[f] != 'object']
quantitive.remove('SalePrice')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

print('Quantitive Data :')
print(quantitive)
print('Qualitative Data :')
print(qualitative)

#%% Add inflation data monthly
def input_cpi():
    cpi = pd.read_csv(INPUT_PATH + 'cpi.csv')
    cpi_temp = []
    for x in train.index:
        for y in cpi.index:
            if train['YrSold'][x] == cpi['Year'][y] and train['MoSold'][x] == cpi['Month'][y]:
                cpi_temp.append(cpi['CPI'][y])
                break
    train['CPI'] = cpi_temp

# Add inflation data annual
def input_annual_cpi():
    cpi = pd.read_csv(INPUT_PATH + 'cpi_annual.csv')
    cpi_temp = []
    for x in train.index:
        for y in cpi.index:
            if train['YrSold'][x] == cpi['Year'][y]:
                cpi_temp.append(cpi['CPI'][y])
                break
    train['CPI'] = cpi_temp

input_cpi()

#%% EDA

# Check Distribution Normal
fig, ax = plt.subplots(figsize=(12,6))
sns.histplot(train['SalePrice'], kde=True)

#%%
# Check Pearson Correlation
fig, ax = plt.subplots(figsize=(30,25))
mat = train.corr('pearson')
mask = np.triu(np.ones_like(mat, dtype=bool))
cmap=sns.diverging_palette(230,20, as_cmap=True)
sns.heatmap(mat, mask=mask, cmap=cmap, vmax=1, center=0, annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
high_corr = pd.DataFrame(mat['SalePrice'][(mat['SalePrice'] > 0.5)])
high_corr.sort_values(by=['SalePrice'], inplace=True, ascending=False)
print(high_corr)

#%%
# SalePrice and OverallQual
plt.figure(figsize=(12,6))
plt.scatter(x=train['OverallQual'], y=train['SalePrice'])
plt.title("SalePrice and OverallQual")
plt.xlabel("OverallQual", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)
plt.show()
figure, ax = plt.subplots(figsize = (12,6))
sns.boxplot(data=train, x = 'OverallQual', y='SalePrice', ax = ax)
plt.show()

# SalePrice and GrLivArea
plt.figure(figsize=(12,6))
plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.title("SalePrice and GrLivArea")
plt.xlabel("GrLivArea", fontsize=13)
plt.ylabel("SalePrice", fontsize=13)
plt.ylim(0,800000)
plt.show()

## SalePrice and Consumer Price Index
# figure, ax = plt.subplots(figsize = (16,4))
# sns.boxplot(data=train, x = 'CPI', y='SalePrice', ax = ax)
# plt.xticks(rotation=45)
# plt.show()

#%% Check dataset's type and count of missing values
#-----Checking Missing Values-----#
# train_test = pd.concat([train, test],axis=0, sort=False)
train_test = train

# Looking at NaN % within the data

nan = pd.DataFrame(train_test.isna().sum(), columns = ['NaN_sum'])
nan['feat'] = nan.index
nan['Perc(%)'] = (nan['NaN_sum']/len(train_test))*100
nan = nan[nan['NaN_sum'] > 0]
nan = nan.sort_values(by = ['NaN_sum'])
nan['Usability'] = np.where(nan['Perc(%)'] > 90, 'Discard', 'Keep')
print(nan)


# Plotting Nan

plt.figure(figsize = (15,5))
sns.barplot(x = nan['feat'], y = nan['Perc(%)'])
plt.xticks(rotation=45)
plt.title('Features containing Nan')
plt.xlabel('Features')
plt.ylabel('% of Missing Data')
plt.show()
#%% Feature engineering
#-----Imputation-----#

fillWithNone = ['PoolQC','MiscFeature', 'Alley', 'Fence', 'MasVnrType', 
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

fillWithZero = ['MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars', 
                'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
                'BsmtFullBath', 'BsmtHalfBath']

for noneValue in fillWithNone:
    train_test[noneValue] = train_test[noneValue].fillna('None')

for zeroValue in fillWithZero:
    train_test[zeroValue] = train_test[zeroValue].fillna(0)

### Special Fill
train_test["LotFrontage"] = train_test.groupby("Neighborhood")["LotFrontage"].transform(
 lambda x: x.fillna(x.median()))
train_test["LotFrontage"] = train_test["LotFrontage"].fillna(train_test["LotFrontage"].mean())
train_test['FireplaceQu'] = train_test['FireplaceQu'].fillna('TA')
train_test['Electrical'] = train_test['Electrical'].fillna("SBrkr")

train_test.isnull().sum()[train_test.isnull().sum()>0]
train_test = train_test.dropna()

#%% Encode categorical label to 0 - n_classes-1
#-----Encoding Categorical Label-----#
from sklearn.preprocessing import LabelEncoder

for col in [f for f in train_test.columns if train_test.dtypes[f] == 'object']:
    labelEncoder = LabelEncoder() 
    labelEncoder.fit(list(train_test[col].values)) 
    train_test[col] = labelEncoder.transform(list(train_test[col].values))

#%%
# Check Pearson Correlation
all_correlation = train_test.corr('pearson')
high_corr = pd.DataFrame()
for key, value in all_correlation.items():
    for x in value.keys():
        corr = all_correlation[key][x]
        if corr > 0.8 and key != x:
            corr_data = {'Key1' : [key], 'Key2': [x], 'Value' : all_correlation[key][x]}
            corr_df = pd.DataFrame(corr_data)
            high_corr = pd.concat([high_corr, corr_df])
high_corr.sort_values(by=['Value'], inplace=True, ascending=False)
print(high_corr)

# #%% Checking correlation on all variables
# positive_correlation = all_correlation[(all_correlation['SalePrice'].values > 0)]
# sf_positive_correlation = positive_correlation.xs('SalePrice', axis=1)
# sf_positive_correlation = sf_positive_correlation.drop(['SalePrice'])
# sf_positive_correlation.plot(kind='barh', title='Variable with positive Correlation', figsize=(6,12))

# #%%
# negative_correlation = all_correlation[(all_correlation['SalePrice'].values <= 0)]
# sf_negative_correlation = negative_correlation.xs('SalePrice', axis=1)
# sf_negative_correlation.plot(kind='barh', title='Variable with negative Correlation', figsize=(6,8))

#%% Check skewness and kurtosis
#-----Checking Skewness and Kurtosis-----#
pd.set_option('display.max_rows', 300)
train_test_skew = pd.DataFrame(train_test.agg(['skew']))
train_test_kurt = pd.DataFrame(train_test.agg(['kurtosis']))
train_test_dist = pd.concat([train_test_skew, train_test_kurt]).transpose()
train_test_dist.sort_values(by=['skew', 'kurtosis'], inplace=True, ascending=False)
print(train_test_dist)

# %% Log transform to reduce skewness
# -----Fixing Skewness-----#
def log_transform(df):
    # Fetch all numeric features
    numeric_features = df.dtypes[df.dtypes != object].index
    skewed_features = df[numeric_features].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_features[skewed_features > 0.5]
    skew_index = high_skew.index

    # Normalize skewed features using log_transformation
        
    for i in skew_index:
        df[i] = np.log1p(df[i])

    return df
train_test = log_transform(train_test)

# %% Drop overfitted data
#-----Dropping Overfitted Data-----#
def overfit_reducer(df):
    overfit = []
    overfit_value = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.5:
            overfit.append(i)
            overfit_value.append(zeros / len(df) * 100)
    overfit = list(overfit)

    for i in overfit:
        df.drop(columns=[i], axis=1, inplace=True)

    print('Overfitted data :\n')
    print(overfit)
    print(overfit_value)
    return df

train_test = overfit_reducer(train_test)

#%%
# Detect Outlier using Cooks Distances

def detect_outlier():
    # storing dependant values
    Y = train_test['SalePrice']
    # storing independent values
    X = train_test.drop(['SalePrice'],axis=1)
    X = sm.add_constant(X)
    # fit the model
    model = sm.OLS(Y, X).fit() 
    np.set_printoptions(suppress=True)
    # create instance of influence
    influence = model.get_influence()
    # get Cook's distance for each observation
    cooks_distances = influence.cooks_distance

    # print Cook's distances
    print(cooks_distances)
    cooks_distances[0][:20]
    threshold = 4 / len(Y)
    # outlier_index = np.where(cooks_distances[0] > (7 * cooks_distances[0].mean()))
    # outlier = cooks_distances[0][cooks_distances[0] > (7 * cooks_distances[0].mean())]
    outlier_index = np.where(cooks_distances[0] > threshold)
    outlier = cooks_distances[0][cooks_distances[0] > threshold]

    # plt.scatter(Y, cooks_distances[0])
    # plt.scatter(Y.iloc[outlier_index[0]], outlier)
    # plt.xlabel('SalePrice')
    # plt.ylabel('Cooks Distance')
    # plt.ylim(0,0.1)
    # plt.show()

    plt.figure(figsize = (12,8))
    plt.scatter(Y.index, cooks_distances[0])
    plt.scatter(outlier_index[0], outlier)
    plt.axhline(threshold, color='r', ls='--', lw=2, label="Cook's Threshold")
    plt.xlabel('Data Index')
    plt.ylabel('Cooks Distance')
    plt.ylim(0,0.01)
    plt.legend(['Threshold = 4 / n', 'Index Data', 'Outlier'], fontsize=10)
    plt.show()

    # plt.figure(figsize=(12,6))
    # plt.scatter(x=train['GrLivArea'], y=train['SalePrice'])
    # plt.scatter(x=train['GrLivArea'][outlier_index[0]], y=train['SalePrice'][outlier_index[0]])
    # plt.title("SalePrice and GrLivArea")
    # plt.xlabel("GrLivArea", fontsize=13)
    # plt.ylabel("SalePrice", fontsize=13)
    # plt.ylim(0,800000)
    # plt.show()

    return cooks_distances
cooks_distances = detect_outlier()
# outliers = np.where(cooks_distances[0] > (3 * cooks_distances[0].mean()))
threshold = 4 / len(train_test['SalePrice'])
outliers = np.where(cooks_distances[0] > threshold)
print("Outlier : ", len(outliers[0]), " data")
train_test = train_test.drop(outliers[0], axis = 0)

# %% Merge data back into train and test

#-----Merging Data-----#
train = pd.get_dummies(train_test)

#----------- Start to make the model----------#
#%% Prepare train data
#-----Preparing Data for Modeling-----#

X = train.drop(['SalePrice'],axis=1)
# X.drop(['CPI'], axis=1, inplace=True)
y = train['SalePrice']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.30, random_state=42)
print('X_train Shape :',X_train.shape)
print('X_test Shape :',X_test.shape)
print('y_train Shape :',y_train.shape)
print('y_test Shape :',y_test.shape)

#%%
#-----Checking Year on Every Data-----#
years = [2006,2007,2008,2009,2010]

for x in range (5):
    temp_train = X_train[(X_train['YrSold']) == years[x]]
    temp_test = X_test[(X_test['YrSold']) == years[x]]
    print("Data train ", years[x], " : ", temp_train.shape)
    print("Data test ", years[x], " : ", temp_test.shape)

# # %% Hyperparameter Tuning with GridSearch
# # -----Hyperparameter Tuning with GridSearch-----#

# hyperparameter_result = []
# model_name = ['LASSO', 'RidgRegression', 'ElasticNet', 'SVR', 'LightGBM']

# class grid():
#     def __init__(self,model):
#         self.model = model
    
#     def grid_get(self,X,y,param_grid):
#         grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error",verbose=10)
#         grid_search.fit(X,y)
#         grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
#         hyperparameter_result.append(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
#         return grid_search

# #%%
# print('Hyperparameter for LASSO')
# GS_lasso = grid(Lasso()).grid_get(X_train,y_train,{'alpha': [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10],'max_iter':[10000]})
# print('Hyperparameter for Ridge Regression')
# GS_ridge = grid(Ridge()).grid_get(X_train,y_train,{'alpha':[1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]})
# print('Hyperparameter for Elastic Net')
# GS_elasticnet = grid(ElasticNet()).grid_get(X_train,y_train,{'alpha':[1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10],'l1_ratio':[0.08,0.1,0.3,0.5],'max_iter':[10000]})
# print('Hyperparameter for SVR')
# GS_svr = grid(SVR()).grid_get(X_train,y_train,{'C' : [0.1, 1, 10, 100], 'epsilon' : [0.01, 0.1, 1], 'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]})
# print('Hyperparameter for LightGBM')
# GS_lgbm = grid(LGBMRegressor()).grid_get(X_train,y_train,{'objective' : ['regression'], 'n_estimators': [500, 1000, 2000], 'max_depth' : [-1, 4, 5], 'learning_rate': [0.001, 0.01, 0.1],  'max_bin': [50, 100, 200]})
# # {'objective' : ['regression'], 'n_estimators': [500, 1000, 2000], 'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],  'max_depth': [4, 5, 6], 'max_bin': [50, 100, 150, 200]}

# #%%
# print(GS_lasso.best_params_, np.sqrt(-GS_lasso.best_score_))
# print(GS_ridge.best_params_, np.sqrt(-GS_ridge.best_score_))
# print(GS_elasticnet.best_params_, np.sqrt(-GS_elasticnet.best_score_))
# print(GS_svr.best_params_, np.sqrt(-GS_svr.best_score_))
# print(GS_lgbm.best_params_, np.sqrt(-GS_lgbm.best_score_))

# #%%
# for hyperparameter, idx in zip(hyperparameter_result, range(5)):
#     with pd.ExcelWriter('revisi_hyperparameter_split_0_30.xlsx',mode='a') as writer:  
#         hyperparameter.to_excel(writer, sheet_name=model_name[idx])

# print(hyperparameter_result)



#%% Training model

#-----Training Model-----#
model_name = ['RidgeRegression', 'LASSO', 'ElasticNet', 'SVR', 'LightGBM']

# Initiate model, variable parameter 0 = with Hyperparameter Tuning, 1 = Default Hyperparameter, 2 = Hyperparameter Tuned
# Use parameter = 2 to run using optimal hyperparameter
def initiate_model(parameter):
    model = []
    if parameter == 0:
        ridge = Ridge(**GS_ridge.best_params_)
        lasso = Lasso(**GS_lasso.best_params_)
        elasticnet = ElasticNet(**GS_elasticnet.best_params_)
        svr = SVR(**GS_svr.best_params_)
        lightgbm = LGBMRegressor(**GS_lgbm.best_params_)
    elif parameter == 1:
        ridge = Ridge()
        lasso = Lasso()
        elasticnet = ElasticNet()
        svr = SVR()
        lightgbm = LGBMRegressor()
    else:
        ridge = Ridge(alpha = 0.01)
        lasso = Lasso(alpha =   0.00001, max_iter = 10000)
        elasticnet = ElasticNet(alpha = 0.0001, l1_ratio = 0.08, max_iter = 10000)
        svr = SVR(C = 10, epsilon = 0.01, gamma = 0.0001, kernel = 'rbf')
        lightgbm = LGBMRegressor(learning_rate = 0.01, max_bin = 50, max_depth = 4, n_estimators = 2000)
    model.append(ridge)
    model.append(lasso)
    model.append(elasticnet) 
    model.append(svr)
    model.append(lightgbm)
    return model
model_cv = initiate_model(2)

#%% Fitting Model
#-----Fitting Model-----#
print('Fitting Model')

model_fitted = {}
training_time = {}

def fit_model():
    for i in range(5):
        start_time= time.time()
        model_fitted[model_name[i]] = model_cv[i].fit(X_train,y_train)
        training_time[model_name[i]] = time.time() - start_time
        print ('Elapsed time for ' + model_name[i] + ' Training : %.2f seconds' % training_time[model_name[i]])

fit_model()

# #%%
# #-----Save Model-----#
# for i in range(5):
#     save_model(model_fitted[model_name[i]], model_name[i])

#%% Print Evaluation Score
#-----Print Evaluation Score-----#
prediction_df = {}
def print_evaluation_score(log):
    if log == True:
            y_test_val = np.expm1(y_test)
    else: y_test_val = y_test
    for i in range(5):
        y_head = model_fitted[model_name[i]].predict(X_test)
        if log == True:
            y_head = np.expm1(y_head)
        prediction_df[model_name[i]] = pd.DataFrame({'Actual': (y_test_val), 'Predicted': (y_head)}).sort_index(axis=0)
        print("'" + '-'*10+ model_name[i] +'-'*10)
        print("'" + 'R square Accuracy: ',r2_score((y_test_val),(y_head)))
        print("'" + 'Mean Absolute Error Accuracy: ',mean_absolute_error((y_test_val),(y_head)))
        print("'" + 'Mean Squared Error Accuracy: ',mean_squared_error((y_test_val),(y_head)))
        print("'" + 'Root Mean Squared Error Accuracy: ', np.sqrt(mean_squared_error((y_test_val),(y_head))))

print_evaluation_score(True)

#%%
# Plotting actual and predicted values
fig = plt.figure(figsize=(15,3))

ax = []
ax.append(fig.add_axes([0.1, 0.5, 0.8, 0.4]))
ax.append(fig.add_axes([0.1, 0.1, 0.8, 0.4]))
ax.append(fig.add_axes([0.1, -0.3, 0.8, 0.4]))
ax.append(fig.add_axes([0.1, -0.7, 0.8, 0.4]))
ax.append(fig.add_axes([0.1, -1.1, 0.8, 0.4]))

ax[0].set_title("House Price Prediction")

for i in range(5):
    ax[0].legend(["Actual","Predicted"],  loc="upper right")
    ax[i].plot(prediction_df[model_name[i]]['Actual'])
    ax[i].plot(prediction_df[model_name[i]]['Predicted'])
    ax[i].set_ylabel(model_name[i])
    ax[i].grid()

# Measuring Inflation Correlation

# #%%
# #-----Clustering with Affinity Propagation-----#
# from sklearn.cluster import AffinityPropagation
# from sklearn import metrics

# cluster_sale_price = train.loc[:,'SalePrice']
# cluster_sale_year = train.loc[:, 'YrSold']
# cluster_sale_month = train.loc[:, 'MoSold']
# cluster_data = train.drop(['SalePrice', 'GarageYrBlt', 'YearBuilt', 'YearRemod/Add', 'YrSold', 'MoSold'], axis=1)
  
# # Compute Affinity Propagation
# af = AffinityPropagation(damping = 0.9, max_iter = 3000, preference=-400000).fit(cluster_data)
# cluster_centers_indices = af.cluster_centers_indices_
# labels = af.labels_
  
# n_clusters_ = len(cluster_centers_indices)

# # #%%
# # def search_best_preference():
# #     preference_number = [-50, -100, -500, -1000, -5000, -10000, -50000, -100000, -150000, -200000, -300000, -400000, -500000]
# #     for x in preference_number:
# #         af = AffinityPropagation(damping = 0.9, max_iter = 3000, preference=x).fit(cluster_data)
# #         print("Cluster with preference : " + str(x) + " is " + str(len(af.cluster_centers_indices_)))
# # search_best_preference()

# #%%
# #-----Correlation CPI and SalePrice using Affinity Propagation-----#
# cluster_data['cluster'] = labels
# cluster_data['SalePrice'] = cluster_sale_price.loc[:]
# cluster_data['YrSold'] = cluster_sale_year.loc[:]

# for x in range (n_clusters_):
#     temp = cluster_data[(cluster_data['cluster'] == x)]
#     corr = temp.corr('pearson')
#     print('label ', x , ': ', corr['SalePrice']['CPI'])

# for label_idx in range(n_clusters_):
#     print("Count data with label " + str(label_idx) + " is " + 
#         str(len(cluster_data[(cluster_data['cluster'] == label_idx)])) +
#         " and SalePrice mean : " + str(np.expm1(cluster_data[(cluster_data['cluster'] == label_idx)].SalePrice.mean()))
#     )

# # %%
# #-----Clustering with K-Means Elbow Method-----#
# from sklearn.cluster import KMeans
# from scipy.spatial.distance import cdist

# cluster_sale_price = train.loc[:,'SalePrice']
# cluster_sale_year = train.loc[:, 'YrSold']
# cluster_sale_month = train.loc[:, 'MoSold']
# cluster_data = train.drop(['SalePrice', 'GarageYrBlt', 'YearBuilt', 'YearRemod/Add', 'YrSold', 'MoSold'], axis=1)

# distortions = []
# inertias = []
# K = range(1, 30)
  
# for k in K:
#     # Building and fitting the model
#     kmeanModel = KMeans(n_clusters=k).fit(cluster_data)
#     kmeanModel.fit(cluster_data)
#     distortions.append(sum(np.min(cdist(cluster_data, kmeanModel.cluster_centers_,
#                                         'euclidean'), axis=1)) / cluster_data.shape[0])
#     inertias.append(kmeanModel.inertia_)

# #%%
# #-----Plotting Elbow Method-----#
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()

# plt.plot(K, inertias, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Inertia')
# plt.title('The Elbow Method using Inertia')
# plt.show()

# #%%
# #-----Clustering with K-Means k=5-----#
# kmean = KMeans(n_clusters=5)
# kmean.fit(cluster_data)

# #%%
# #-----Correlation CPI and SalePrice using K-Means-----#
# cluster_data['cluster'] = pd.DataFrame(kmean.labels_).loc[:, 0]
# cluster_data['SalePrice'] = cluster_sale_price.loc[:]
# cluster_data['YrSold'] = cluster_sale_year.loc[:]
# for x in range (5):
#     temp = cluster_data[(cluster_data['cluster'] == x)]
#     corr = temp.drop(columns=['cluster'], axis=1).corr('pearson')
#     print('label ', x , ': ', corr['SalePrice']['CPI'])
# for label_idx in range(5):
#     print("Count data with label " + str(label_idx) + " is " + 
#         str(len(cluster_data[(cluster_data['cluster'] == label_idx)])) +
#         " and SalePrice mean : " + str(np.expm1(cluster_data[(cluster_data['cluster'] == label_idx)].SalePrice.mean()))
#     )



# # %%
# year_col = [2006, 2007, 2008, 2009, 2010]
# for label_idx in range(5):
#     for year in year_col:
#         print('Label ' + str(label_idx) + ' SalePrice mean value at ' + str(year) + 
#         ' is ' + str(np.expm1(cluster_data[(cluster_data['cluster'] == label_idx) & (cluster_data['YrSold'] == year)].mean()['SalePrice'])))
        

# #%%
# cpi_data = pd.read_csv(INPUT_PATH + 'cpi_annual.csv')
# str_year = [str(i) for i in year_col]
# cluster_dict = []
# for label_idx in range(5):
#     temp = {}
#     for year in year_col:
#         temp[year] = np.expm1(cluster_data[(cluster_data['cluster'] == label_idx) & (cluster_data['YrSold'] == year)].mean()['SalePrice'])
#     cluster_dict.append(temp)

# figure, axis = plt.subplots(figsize = (12,6))
# axis.plot(str_year, cpi_data['CPI'], color='lime', label='CPI')
# axis.set_title("CPI growth by year")
# axis.set_xlabel("Year")
# axis.set_ylabel("CPI")

# # Plotting both the curves simultaneously
# figure, axis = plt.subplots(figsize = (12,7))
# colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'darkcyan']
# for label_idx, color in zip(range(5), colors):
#     axis.plot(str_year, list(cluster_dict[label_idx].values()), color=color, label='cluster_' + str(label_idx))

# # Naming the x-axis, y-axis and the whole graph
# axis.set_title("SalePrice growth by year")
# axis.set_xlabel("Year")
# axis.set_ylabel("SalePrice")
  
# # Adding legend, which helps us recognize the curve according to it's color
# figure.legend(loc=7)
# figure.tight_layout()
# figure.subplots_adjust(right=0.85)
  
# # To load the display window
# plt.show()
# %%

