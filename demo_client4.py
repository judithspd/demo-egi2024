import flwr as fl
import numpy as np
from pathlib import Path

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn import metrics

import certifi

import ai4flwr.auth.bearer


token = ""


# Load and process data:
x_client4 = np.fromfile('./data/x_client4.npy').reshape((16, 150, 150))
y_client4 = np.load('./data/y_client4.npy')
x_client4 = np.reshape(x_client4, (16, 150, 150, 1))

x_train, x_test, y_train, y_test = train_test_split(x_client4, y_client4,
                                                    test_size=0.25, random_state=42, stratify=y_client4)

# Model to be trained:
model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(128 , activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()

# Flower client
class Client4(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5, batch_size=16)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


auth_plugin = ai4flwr.auth.bearer.BearerTokenAuthPlugin(token)

# Start -> connecting with the server
uuid = "" # Fill with the UUID of the deployment of the FL server (AI4EOSC)
end_point = f"fedserver-{uuid}.iisas-deployments.cloud.ai4eosc.eu"
fl.client.start_client(
    server_address=f"{end_point}:443", 
    client=Client4().to_client(),
    root_certificates=Path(certifi.where()).read_bytes(),
    call_credentials=auth_plugin.call_credentials()
)

model.fit(x_train, y_train, epochs=1, batch_size=16)

score = model.evaluate(x_test, y_test)
pred = model.predict(x_test)
fpr, tpr, _ = metrics.roc_curve(y_test, pred)
auc = metrics.auc(fpr, tpr)
print(f'CLIENT 4: Test loss: {score[0]} / Test accuracy: {score[1]} / Test AUC: {auc}')



