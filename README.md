# Распознавание эмоций на групповых фотографиях

## Задача
Требуется разработать компьютерную программу для распознавания эмоции группового фото, используя методы deep learning.
Программа принимает на вход изображение, а на выходе делает вывод о том какая из трех эмоций (positive, neutral, negative) изображена на фото.

## Требования

- python 3.6.1

- numpy 1.13.1

- opencv 3.2.0

Необходимо поместить в папку Resourses:

- Emotion Recognition Caffe model (EmotiW_VGG_S.caffemodel) - https://gist.github.com/GilLevi/54aee1b8b0397721aa4b

- Emotion Recognition Caffe model configuration - https://gist.github.com/GilLevi/54aee1b8b0397721aa4b#file-deploy-txt

#### Запустить программу можно, выполнив следующую команду

run.py <путь до изображения>

## Архитектура

Программа состоит из 3х основных этапов:

1. Распознвание лиц на изображении

Для распознавания лиц людей в работе используется предобученный классификатор (по фичам Хаара) доступный в opencv как cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

2. Распознавание эмоций по каждому найденному лицу

Для классификации эмоций по выражению лица используется сверточная нейронная сеть, описанная в работе http://www.openu.ac.il/home/hassner/projects/cnn_emotions/ . Выделяем 7 эмоций : angry, disgust, fear, happy, neutral, sad, surprise. Для каждого лица на фото программа определяет с каким весом каждая эмоция к нему относится (чем больше вес, тем больше вероятность). Эмоция с максимальным весом является результатом.

3. Определение общей эмоции фотографии

Для определения общей эмоции фото, складываются все веса эмоций для всех найденных лиц, эмоция с максимальным весом является результатом. Для того чтобы классифицировать фото по трем категориям - positive, nеutral, negative мы отнесли happy, surprise к *positive*, neutral к *neutral*, angry, disgust, fear, sad к *negative*

## Презентация

https://docs.google.com/presentation/d/1prYpLmHk5XnzH5Ixc6_IUmtokMvVokNMRMOVkLVEt00/edit#slide=id.g25cf51c1fc_1_0





