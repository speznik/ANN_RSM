import numpy as np
import pandas as pd
from keras.models import model_from_json
import os
from keras import losses,metrics
import tensorflow as tf

# load json and create model
json_file = open('/ANN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/ANN_model1.h5")
print("Loaded model from disk")

#loading test data
data1 = pd.read_csv('/content/new_data.csv')
dataset = data1.values

r_index = []
for i in range(29):
	r_index.append(i)

ferm_time = []
msg = []
sp_cont = []
glucose_cont = []
folate_cont = []

for i in range(len(r_index)):
	ferm_time.append(dataset[r_index[i],0])
	msg.append(dataset[r_index[i],1])
	sp_cont.append(dataset[r_index[i],2])
	glucose_cont.append(dataset[r_index[i],3])
	folate_cont.append(dataset[r_index[i],4])

X = dataset[:,0:4]
Y = list(dataset[:,4])
# y_actual = list(dataset[:,3])

Yout = loaded_model.predict(X)

error = []
Ypred1 = Yout.tolist()
Ypred = []

for i in range(len(Y)):
	Ypred.append(round(Ypred1[i][0],4))
	error.append(np.absolute(Ypred[i]-Y[i]))

C = {'Fermentation Time (hrs)': ferm_time,
        'Monosodium Glutamate (micromolar/L)': msg,
        'Orange-fleshed Sweet Potato Content (%)': sp_cont,
		'Glucose Content (%)':glucose_cont,
        'Folate content (mcg/100g)': Y,
        'Folate content (mcg/100g)(predicted)': Ypred,
        'Net_Error(absolute)':error
    }

df = pd.DataFrame(C, columns= ['Fermentation Time (hrs)', 'Monosodium Glutamate (micromolar/L)', 'Orange-fleshed Sweet Potato Content (%)', 'Glucose Content (%)', 'Folate content (mcg/100g)', 'Folate content (mcg/100g)(predicted)', 'Net_Error(absolute)'])
export_csv = df.to_csv (r'/pred_model_new.csv', index = None, header=True)
print (df)
