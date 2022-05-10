from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
lr = LogisticRegression()
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)
plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:3], digits.target[0:3])):
    plt.subplot(1, 3, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)
    plt.title('Training: %i\n' % label, fontsize=20)
lr.fit(x_train, y_train)
predictions = lr.predict(x_test)
score = lr.score(x_test, y_test)
print("Score: %.3f" % score)
