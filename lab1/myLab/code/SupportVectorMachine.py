import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def read_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))


def load_dataset(label_dict):
    train_X = read_data('C:\\Users\\Zlat\\Desktop\\Прикладной_ии\\lab1\\myLab\\code', 'train.csv').values[:,:-2]
    train_y = read_data('C:\\Users\\Zlat\\Desktop\\Прикладной_ии\\lab1\\myLab\\code', 'train.csv')['Activity']
    train_y = train_y.map(label_dict).values
    test_X = read_data('C:\\Users\\Zlat\\Desktop\\Прикладной_ии\\lab1\\myLab\\code', 'test.csv').values[:,:-2]
    test_y = read_data('C:\\Users\\Zlat\\Desktop\\Прикладной_ии\\lab1\\myLab\\code', 'test.csv')
    test_y = test_y['Activity'].map(label_dict).values
    return train_X, train_y, test_X, test_y


label_dict = {'WALKING': 0, 'WALKING_UPSTAIRS': 1, 'WALKING_DOWNSTAIRS': 2, 'SITTING':3, 'STANDING': 4, 'LAYING': 5}
train_X, train_y, test_X, test_y = load_dataset(label_dict)


target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
f1 = []
parameters = [1, 2, 3, 4]
kernels = ["linear", "rbf", "sigmoid", "poly"]
# for kernel in kernels:
#     model = SVC(kernel=kernel)
#     model.fit(train_X, train_y)
#     yhat = model.predict(test_X)
#     res = classification_report(test_y, yhat, target_names=target_names, output_dict=True)
#     f1.append(round(sum([res.get(target_name).get("f1-score") for target_name in target_names]) / len(target_names), 2))

model = SVC(kernel="linear")
model.fit(train_X, train_y)
yhat = model.predict(test_X)
print(classification_report(test_y, yhat, target_names=target_names))

# plt.plot(parameters, f1, 'o-r', color='g')
# plt.text(1, 0.88, "1 point - linear \n2 point - rbf \n3 point - sigmoid \n4 point - polynomial", color="r")
# plt.xlabel('kernel')
# plt.ylabel('f1')
# plt.title('RandomForestClassifier')
# plt.grid(True)
# plt.show()
