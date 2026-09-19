"""
Trains a CNN (MobileNetV2 transfer learning) to classify skin rash images
into Measles vs lookalike diseases, using the MSLD v2.0 dataset (fold1).
Handles class imbalance via class weights + heavy augmentation.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "data/raw/MSLD_v2/Original Images/Original Images/FOLDS/fold1"
TRAIN_DIR = os.path.join(BASE, "Train")
VALID_DIR = os.path.join(BASE, "Valid")
TEST_DIR = os.path.join(BASE, "Test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 25

os.makedirs("models/image_classifier", exist_ok=True)
os.makedirs("data/processed/eda_plots", exist_ok=True)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=True, seed=42,
)
valid_gen = eval_datagen.flow_from_directory(
    VALID_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False,
)
test_gen = eval_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="categorical", shuffle=False,
)

class_names = list(train_gen.class_indices.keys())
print(f"Classes: {class_names}")
print(f"Train samples: {train_gen.samples}, Valid: {valid_gen.samples}, Test: {test_gen.samples}")

class_weights_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes,
)
class_weights = dict(enumerate(class_weights_arr))
print(f"Class weights: {dict(zip(class_names, class_weights_arr.round(2)))}")

base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(len(class_names), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ModelCheckpoint("models/image_classifier/best_model.keras", monitor="val_accuracy", save_best_only=True),
]

print("\n=== Phase 1: training classifier head ===")
history1 = model.fit(
    train_gen, validation_data=valid_gen, epochs=EPOCHS,
    class_weight=class_weights, callbacks=callbacks, verbose=1,
)

print("\n=== Phase 2: fine-tuning top layers ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])

history2 = model.fit(
    train_gen, validation_data=valid_gen, epochs=15,
    class_weight=class_weights, callbacks=callbacks, verbose=1,
)

test_gen.reset()
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

print("\n" + "=" * 60)
print("TEST SET CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=class_names))

report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
if "Measles" in report_dict:
    print(f"\n>>> MEASLES RECALL: {report_dict['Measles']['recall']:.3f}")
    print(f">>> MEASLES PRECISION: {report_dict['Measles']['precision']:.3f}")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Image Classifier (Test Set)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/image_confusion_matrix.png", dpi=120)
plt.close()

model.save("models/image_classifier/final_model.keras")
with open("models/image_classifier/class_names.txt", "w") as f:
    f.write("\n".join(class_names))

print("\nSaved model to models/image_classifier/final_model.keras")
print("Saved confusion matrix to data/processed/eda_plots/image_confusion_matrix.png")
