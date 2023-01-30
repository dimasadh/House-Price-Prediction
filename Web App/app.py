from flask import Flask, request, render_template, redirect, url_for, flash
import pickle
import price_predict

MODEL_PATH = "../Model/"

model_name = ['RidgeRegression.pkl', 'LASSO.pkl', 'ElasticNet.pkl', 'SVR.pkl', 'LightGBM.pkl']
models = []
# Use pickle to load in the pre-trained model.
for model in model_name:
    with open(MODEL_PATH + model, 'rb') as f:
        models.append(pickle.load(f))
print(models)

app = Flask(__name__)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.form
    # Membuat dictionary dari data yang diterima
    input_data = {
        'MSSubClass': data['MSSubClass'],
        'MSZoning': data['MSZoning'],
        'LotFrontage': data['LotFrontage'],
        'LotArea': data['LotArea'],
        'Street': data['Street'],
        'Alley': data['Alley'],
        'LotShape': data['LotShape'],
        'LandContour': data['LandContour'],
        'Utilities': data['Utilities'],
        'LotConfig': data['LotConfig'],
        'LandSlope': data['LandSlope'],
        'Neighborhood': data['Neighborhood'],
        'Condition1': data['Condition1'],
        'Condition2': data['Condition2'],
        'BldgType': data['BldgType'],
        'HouseStyle': data['HouseStyle'],
        'OverallQual': data['OverallQual'],
        'OverallCond': data['OverallCond'],
        'YearBuilt': data['YearBuilt'],
        'YearRemod/Add': data['YearRemod/Add'],
        'RoofStyle': data['RoofStyle'],
        'RoofMatl': data['RoofMatl'],
        'Exterior1st': data['Exterior1st'],
        'Exterior2nd': data['Exterior2nd'],
        'MasVnrType': data['MasVnrType'],
        'MasVnrArea': data['MasVnrArea'],
        'ExterQual': data['ExterQual'],
        'ExterCond': data['ExterCond'],
        'Foundation': data['Foundation'],
        'BsmtQual': data['BsmtQual'],
        'BsmtCond': data['BsmtCond'],
        'BsmtExposure': data['BsmtExposure'],
        'BsmtFinType1': data['BsmtFinType1'],
        'BsmtFinSF1': data['BsmtFinSF1'],
        'BsmtFinType2': data['BsmtFinType2'],
        'BsmtFinSF2': data['BsmtFinSF2'],
        'BsmtUnfSF': data['BsmtUnfSF'],
        'TotalBsmtSF': data['TotalBsmtSF'],
        'Heating': data['Heating'],
        'HeatingQC': data['HeatingQC'],
        'CentralAir': data['CentralAir'],
        'Electrical': data['Electrical'],
        '1stFlrSF': data['1stFlrSF'],
        '2ndFlrSF': data['2ndFlrSF'],
        'LowQualFinSF': data['LowQualFinSF'],
        'GrLivArea': data['GrLivArea'],
        'BsmtFullBath': data['BsmtFullBath'],
        'BsmtHalfBath': data['BsmtHalfBath'],
        'FullBath': data['FullBath'],
        'HalfBath': data['HalfBath'],
        'BedroomAbvGr': data['BedroomAbvGr'],
        'KitchenAbvGr': data['KitchenAbvGr'],
        'KitchenQual': data['KitchenQual'],
        'TotRmsAbvGrd': data['TotRmsAbvGrd'],
        'Functional': data['Functional'],
        'Fireplaces': data['Fireplaces'],
        'FireplaceQu': data['FireplaceQu'],
        'GarageType': data['GarageType'],
        'GarageYrBlt': data['GarageYrBlt'],
        'GarageFinish': data['GarageFinish'],
        'GarageCars': data['GarageCars'],
        'GarageArea': data['GarageArea'],
        'GarageQual': data['GarageQual'],
        'GarageCond': data['GarageCond'],
        'PavedDrive': data['PavedDrive'],
        'WoodDeckSF': data['WoodDeckSF'],
        'OpenPorchSF': data['OpenPorchSF'],
        'EnclosedPorch': data['EnclosedPorch'],
        '3SsnPorch': data['3SsnPorch'],
        'ScreenPorch': data['ScreenPorch'],
        'PoolArea': data['PoolArea'],
        'PoolQC': data['PoolQC'],
        'Fence': data['Fence'],
        'MiscFeature': data['MiscFeature'],
        'MiscVal': data['MiscVal'],
        'MoSold': data['MoSold'],
        'YrSold': data['YrSold'],
        'SaleType': data['SaleType'],
        'SaleCondition': data['SaleCondition'],
        'SalePrice': data['SalePrice']
    }

    results = price_predict.predict(models, input_data)
    names = ['Ridge Regression', 'LASSO', 'Elastic Net', 'Support Vector Regression', 'LightGBM']
    return render_template('result.html', results = results, model_name = names)

if __name__ == '__main__':
    app.run()
    #flask run --host=0.0.0.0