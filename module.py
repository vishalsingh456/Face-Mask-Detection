from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os


INIT_LR = 1e-4
EPOCHS = 50
BS = 32

DIRECTORY = r"E:\clj internship\clj\face_detection\data set"
CATEGORIES = ['with mask', 'without mask']


data = []
Labels = []

print("[INFO] Differentiating Images......")
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for image in os.listdir(path):
        img_path = os.path.join(path, image)
        img = load_img(img_path, target_size=(224 ,224) )
        img = img_to_array(img)
        img = preprocess_input(img)

        data.append(img)
        Labels.append(category)


print("[INFO] Generating Image Data.....")
aug = ImageDataGenerator(
	rotation_range=40,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


lb = LabelBinarizer()
Labels = lb.fit_transform(Labels)
Labels = to_categorical(Labels)

print("[INFO] Converting The Data Set Into Array......")
data = np.array(data, dtype= 'float32')
Labels = np.array(Labels)
print(Labels.shape)
print(Labels.size)


(x_train, x_test, y_train, y_test) = train_test_split(data, Labels, test_size=0.20, stratify=Labels, random_state=42)

print("[INFO] Creating BaseModel....")
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

print("[INFO] Creating HeadModel.....")
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs = baseModel.input, outputs = headModel)

print("[INFO] Deleting Pretrain Data.......")
for layer in baseModel.layers:
    layer.trainable = False

print('[INFO] Compiling Model.....')
opt = Adam(lr = INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training head........")
H = model.fit(aug.flow(x_train , y_train, batch_size = BS),\
     steps_per_epoch=len(x_train)//BS,validation_data = (x_test,y_test),\
        validation_steps = len(x_test)//BS, epochs = EPOCHS)

print("[INFO] evaluating network......")
pred_Index = model.predict(x_test, batch_size=BS)

pred_Index = np.argmax(pred_Index, axis = 1)

print(classification_report(y_test.argmax(axis = 1), pred_Index, target_names = lb.classes_))

print('[INFO] saving mask detector model...' )
model.save("Mask_Detector.model", save_format = "h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCHS), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0,EPOCHS), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0,EPOCHS), H.history["accuracy"], label = "train_accuracy")
plt.plot(np.arange(0,EPOCHS), H.history["val_accuracy"], label = "val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("Loss_Acc.png")