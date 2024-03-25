import numpy as np
import cv2
from tensorflow import keras
import os
import matplotlib.pyplot as plt

model = keras.models.load_model('best_trained.h5')
classes = ['aca cancer', 'normal', 'scc cancer']

def preprocessing(image_path, size=(256, 256)):
    img = cv2.imread(image_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
    img = cv2.resize(img, size)
    img = img/255.0
    return img
def display(img_path, number):
    plt.figure(figsize=(15, 5))
    for i in range(number):
        plt.subplot(1, number, i+1)
        img = plt.imread(img_path[i])
        plt.imshow(img)
        plt.axis('off')
    plt.show()

path = 'lung_colon_image_set/lung_image_sets/lung_scc'
filenames = os.listdir(path)
i, j, k, l = 0, 0, 0, 0
aca_list = [''] * 500
n_list = [''] * 500
unclassified = [''] * 500
for file in filenames:
    wrong_file = file
    image_path = os.path.join(path, file)   
    ip_img = preprocessing(image_path)
    predictions = model.predict(np.expand_dims(ip_img, axis=0))
    predict_classes_ix = np.argmax(predictions)
    predict_class = classes[predict_classes_ix]
    # print("It seems like that is -", predict_class)
    if predict_class == classes[2]:
        i += 1
    elif predict_class == classes[0]:
        aca_list[j] = wrong_file
        j += 1
    elif predict_class == classes[1]:
        n_list[k] = wrong_file
        k += 1
    else:
        unclassified[l] = wrong_file
    print("Roll: ", i)

print('best_trained.h5')
print('Correct predictions of SCC Cancer: ', i)
print('Incorrect predictions of SCC Cancer: ', (5000-i))
print('Predicted ACA Cancer:', j)
print('Predicted Normal Lung:', k)
print('Unclassified:', l)
np.savetxt('best_trained/scc_run/aca_list.txt', aca_list, fmt='%s')
np.savetxt('best_trained/scc_run/n_list.txt', n_list, fmt='%s')
np.savetxt('best_trained/scc_run/unclassified.txt', unclassified, fmt='%s')

