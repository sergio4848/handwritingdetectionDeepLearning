import tensorflow as tf

# MNIST veri kümesini yükleme
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Veri setini normalize etme
train_images = train_images / 255.0
test_images = test_images / 255.0

# Model oluşturma
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Modeli eğitme
model.fit(train_images.reshape(-1, 28, 28, 1), train_labels, epochs=5)

# Modelin performansını değerlendirme
test_loss, test_acc = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)
