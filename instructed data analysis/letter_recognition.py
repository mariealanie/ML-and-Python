import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow_datasets as tfds  


print("Загрузка выборки EMNIST (буквы)...")
ds_train, ds_test = tfds.load('emnist/letters', split=['train', 'test'], as_supervised=True)


x_train, y_train = [], []
x_test, y_test = [], []

for img, label in tfds.as_numpy(ds_train):
    x_train.append(img)
    y_train.append(label)
for img, label in tfds.as_numpy(ds_test):
    x_test.append(img)
    y_test.append(label)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


y_train -= 1
y_test -= 1


plt.figure(figsize=(10,5))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')   
])

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train_cat = keras.utils.to_categorical(y_train, 26)
y_test_cat = keras.utils.to_categorical(y_test, 26)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Обучение модели...")
model.fit(x_train, y_train_cat, batch_size=64, epochs=10, validation_split=0.2, verbose=1)

print('******* loss и accuracy на тестовом наборе ***********')
model.evaluate(x_test, y_test_cat)


n = 5
x = np.expand_dims(x_test[n], axis=0)
res = model.predict(x)
pred = np.argmax(res)

print(f"Модель распознала: {chr(pred + 97)} (должна быть {chr(y_test[n] + 97)})")  
plt.imshow(x_test[n], cmap=plt.cm.binary)
plt.show()

pred_all = np.argmax(model.predict(x_test), axis=1)
print('Ошибочные результаты:')
er = 0
for i in range(10000):
    if pred_all[i] != y_test[i]:
        er += 1
        if 10 < er < 16:
            print(f"Истинно: {chr(y_test[i]+97)}, предсказано: {chr(pred_all[i]+97)}")
            plt.imshow(x_test[i], cmap=plt.cm.binary)
            plt.show()

print('Количество ошибок:', er)

