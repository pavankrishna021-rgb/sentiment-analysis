import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import json
import numpy as np

print("Loading IMDB dataset...")
vocab_size = 10000
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

max_length = 200
X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post', truncating='post')

print("Building model...")
model = models.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Training model...")
model.fit(X_train_padded, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

test_loss, test_acc = model.evaluate(X_test_padded, y_test)
print(f"Test accuracy: {test_acc:.4f}")

model.save('sentiment_model.keras')
print("Model saved!")

word_index = tf.keras.datasets.imdb.get_word_index()
with open('word_index.json', 'w') as f:
    json.dump(word_index, f)
print("Word index saved!")
print("DONE! Ready for deployment.")
