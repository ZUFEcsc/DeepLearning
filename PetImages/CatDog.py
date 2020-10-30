#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import layers
import warnings
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings('ignore')

#from keras.optimizers import SGD



#数据准备
trainGen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rotation_range=20,
    #width_shift_range=0.2,
    #height_shift_range=0.2,
    brightness_range=(0.7,1.3),
    zoom_range=0.3,
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True)
evalGen = ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    rescale=1./255)
csv_logger = CSVLogger('training.csv')
trainPath = 'train'
evalPath = 'test'

#构建模型
input_image_shape = (100,100,3)
base_model = InceptionV3(input_shape=input_image_shape,
                         weights='imagenet',
                         include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#训练
model.fit_generator(trainGen.flow_from_directory(trainPath,
                                                 target_size=(100, 100),
                                                 #save_to_dir='what',
                                                 batch_size=2),
                    steps_per_epoch=331, workers=8,
                    validation_data=evalGen.flow_from_directory(evalPath,
                                                 target_size=(100, 100),
                                                 batch_size=4),
                    validation_steps=544,
                    callbacks=[csv_logger],
                    verbose=2, epochs=2)

scores = model.evaluate_generator(evalGen.flow_from_directory(evalPath,
                                                 target_size=(100, 100),
                                                 batch_size=4),
                     steps=544,
                     verbose=2)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

model.save('CatDog.h5')

# input('Press any key......')
