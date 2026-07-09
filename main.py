import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = []
labels = []

base_path = "PetImages"

for label, folder in enumerate(["Cat", "Dog"]):
    folder_path = os.path.join(base_path, folder)

    count = 0
    for file in os.listdir(folder_path):
        try:
            img_path = os.path.join(folder_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            data.append(img.flatten())
            labels.append(label)

            count += 1
            if count == 500:
                break

        except:
            pass

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

sample = X_test[0].reshape(64, 64)

plt.imshow(sample, cmap="gray")
plt.title("Prediction: " + ("Cat" if y_pred[0] == 0 else "Dog"))
plt.axis("off")
plt.show()