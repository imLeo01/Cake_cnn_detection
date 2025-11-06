import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# =========================
# 1. ĐƯỜNG DẪN DATA
# =========================
train_dir = r"C:\Users\Lazycat\Downloads\data\train"
val_dir   = r"C:\Users\Lazycat\Downloads\data\valid"

IMG_SIZE   = (180, 180)
BATCH_SIZE = 32
EPOCHS     = 40 

# =========================
# 2. TẠO DATASET
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
num_classes = len(class_names)
print("🔹 Class Names:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)

# Đổi label sang one-hot để dùng CategoricalCrossentropy + label smoothing
train_ds = train_ds.map(
    lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
    num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.map(
    lambda x, y: (x, tf.one_hot(y, depth=num_classes)),
    num_parallel_calls=AUTOTUNE
)

# =========================
# 3. DATA AUGMENTATION
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.05, 0.05),
])

# =========================
# 4. MODEL CNN FROM SCRATCH
# =========================
inputs = layers.Input(shape=IMG_SIZE + (3,))

x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)

x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D()(x)

x = layers.Dropout(0.4)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, outputs)

# =========================
# 5. COMPILE (label smoothing)
# =========================
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.summary()

# =========================
# 6. CALLBACKS
# =========================
checkpoint_cb = callbacks.ModelCheckpoint(
    "best_cake_model.h5",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

earlystop_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

# =========================
# 7. TRAIN
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

# Lưu model cuối (đã là best vì restore_best_weights=True)
model.save("bestcake.h5")
print("✅ Đã lưu model tại bestcake.h5 ")
