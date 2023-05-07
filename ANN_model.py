import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import metrics
from keras.models import model_from_json

data1 = pd.read_csv('/content/new_data.csv')
dataset = data1.values

r_index = np.random.permutation(21)

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

for i in range(len(r_index)):
    dataset[i,0] = ferm_time[i]
    dataset[i,1] = msg[i]
    dataset[i,2] = sp_cont[i]
    dataset[i,3] = glucose_cont[i]
    dataset[i,4] = folate_cont[i]

# split into input (X) and output (Y) variables
X = dataset[:,0:4]
Y = dataset[:,4]

model = Sequential()
model.add(Dense(1000, input_dim=4, kernel_initializer='normal', activation='sigmoid')) # epoch number can be changed
model.add(Dense(250, activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation="linear"))
model.compile(loss=losses.logcosh, optimizer='adam', metrics = [metrics.logcosh])

results = model.fit(X, Y, validation_split=0.2, epochs=1000, verbose=0)
scores = model.evaluate(X, Y, verbose=0)

print("Test-Accuracy:", np.mean(results.history["val_loss"]))

print("%s: %f" % (model.metrics_names[1], scores[1]))

# serialize model to JSON
model_json = model.to_json()
with open("/ANN_model.json", "w") as json_file: # Replace with different models
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/ANN_model1.h5") # Replace with different models
print("Saved model to disk")
