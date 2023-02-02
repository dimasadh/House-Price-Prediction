#%% 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

INPUT_PATH = "../Datasets/"

# Use pickle to load in the pre-trained model.
with open(f'../Model/LightGBM.pkl', 'rb') as f:
    model = pickle.load(f)

def encode_variable(dataToEncode):
    all_data = pd.read_csv(INPUT_PATH + 'AmesHousing.csv')
    all_data.columns = all_data.columns.str.replace(' ', '')

    colNumeric = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 
                'OverallCond', 'YearBuilt', 'YearRemod/Add', 'MasVnrArea', 'BsmtFinSF1', 
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
                'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
                'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 
                'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', \
                '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    
    for col in colNumeric:
        dataToEncode[col] = np.float64(dataToEncode[col])

    all_data = all_data.append(dataToEncode, ignore_index=True)

    colToEncode = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
        'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 
        'Exterior2nd','MasVnrType', 'ExterQual','ExterCond', 'Foundation', 
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
        'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

    for col in colToEncode:
        labelEncoder = LabelEncoder() 
        labelEncoder.fit(list(all_data[col].values)) 
        all_data[col] = labelEncoder.transform(list(all_data[col].values))

    ## drop unused variable
    variableToDrop = ['Order', 'PID', 'Street', 'Utilities', 'PoolArea', 'PoolQC', 'SalePrice']
    for i in variableToDrop:
        all_data.drop(columns=[i], axis=1, inplace=True)

    return all_data.loc[len(all_data)-1]


def preprocess(data):
    ## imputation
    fillWithNone = ['MiscFeature', 'Alley', 'Fence', 'MasVnrType', 
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

    fillWithZero = ['MasVnrArea', 'GarageYrBlt', 'GarageArea', 'GarageCars', 
                    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 
                    'BsmtFullBath', 'BsmtHalfBath', 'LotFrontage']

    for noneValue in fillWithNone:
        if data[noneValue] == None or data[noneValue] == 'nan' :
            data[noneValue] = 'None'

    for zeroValue in fillWithZero:
        if data[zeroValue] == None or data[zeroValue] == 'nan':
            data[zeroValue] = 0

    ### Special Fill
    if data['FireplaceQu'] == None:
        data['FireplaceQu'] = 'TA'
    if data['Electrical'] == None:
        data['Electrical'] = "SBrkr"

    ## encoding variabel
    data = encode_variable(data)

    ## input cpi
    cpi = pd.read_csv(INPUT_PATH + 'cpi.csv')
    for y in cpi.index:
        if data['YrSold'] == cpi['Year'][y] and data['MoSold'] == cpi['Month'][y]:
            data['CPI'] = cpi['CPI'][y]
            break

    ## log transformation
    variableToTransform = ['MiscVal', 'LotArea', 'LowQualFinSF',
       'Heating', 'Condition2', '3SsnPorch', 'RoofMatl', 'LandSlope',
       'MiscFeature', 'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch',
       'ScreenPorch', 'BsmtHalfBath', 'Condition1', 'MasVnrArea',
       'OpenPorchSF', 'BldgType', 'WoodDeckSF', 'RoofStyle',
       'RoofStyle', 'LotFrontage', '1stFlrSF', 'BsmtFinSF1', 'MSSubClass', 
       'GrLivArea', 'TotalBsmtSF', 'BsmtUnfSF',
       '2ndFlrSF', 'TotRmsAbvGrd', 'Fireplaces', 'HalfBath', 'GarageType',
       'BsmtFullBath', 'OverallCond']

    # Normalize skewed features using log_transformation
    for i in variableToTransform:
        data[i] = np.log1p(data[i])
    return data

def predict(models, data):
    results = {}
    model_names = ['Ridge Regression', 'LASSO', 'Elastic Net', 'Support Vector Regression', 'LightGBM']

    cleanData = preprocess(data)
    cleanData = np.array(cleanData).reshape(1, -1)

    for i, model in enumerate(models):
        prediction = np.expm1(model.predict(cleanData)[0])
        results[model_names[i]] = round(prediction, 2)
        # results.append(np.expm1(model.predict(cleanData)))

    print(results)
    return results

# #%% Testing
# #-----Getting Dataset-----#
# raw_data = pd.read_csv(INPUT_PATH + 'AmesHousing.csv')
# raw_data.columns = raw_data.columns.str.replace(' ', '')
# test = predict(model, raw_data.loc[0])
# %%
