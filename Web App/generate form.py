#%%
import pandas as pd

INPUT_PATH = "../Datasets/"

data = pd.read_csv(INPUT_PATH + 'AmesHousing.csv')
data.columns = data.columns.str.replace(' ', '')
print(data)
# %%
for x in data.keys():
    val = data.loc[0,x]
    if data[x].dtypes == 'object':
        data_type = "text"
    else:
        data_type = "number"
    print("<label for='"+x+"'>"+x+":</label> \n<input type='"+data_type+"' id='"+x+"' value=", val, " name='"+x+"'><br>\n")
# %%
for x in data.keys():
    val = data.loc[0,x]
    if data[x].dtypes == 'object':
        data_type = "text"
    else:
        data_type = "number"
    print("'"+x+"': data['"+x+"'],")

# %%
