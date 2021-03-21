import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils.vis_utils import plot_model

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os



train_dir = "./TRAIN"
val_dir = "./TEST"

        

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                                width_shift_range=0.2,
                                                                height_shift_range=0.2, 
                                                                shear_range=0.2,
                                                                zoom_range=0.2,
                                                                horizontal_flip=True,
                                                                rescale=1./255)
#split test data to validation and tesing 
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                              validation_split=0.5)



train_set = train_datagen.flow_from_directory(train_dir, class_mode = 'binary',
                                              batch_size = 32, 
                                              target_size=(65,65))

val_set = val_datagen.flow_from_directory(val_dir, class_mode = 'binary',
                                              batch_size = 32, 
                                              target_size=(65,65),
                                              subset= 'training')

test_set = val_datagen.flow_from_directory(val_dir, class_mode = 'binary',
                                              batch_size = 32, 
                                              target_size=(65,65),
                                              subset= 'validation')


# define convolutional block

# In[3]:


def conv_block(inputs, filters):
    x = tf.keras.layers.Conv2D(filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    return x


# create the model

# In[4]:


def create_model():
    inputs = tf.keras.Input(shape=(65, 65, 3))
    
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = conv_block(x, 256)

    
    x = tf.keras.layers.Flatten()(x)
    
    x = tf.keras.layers.Dense(4096, activation = 'relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
        
    x = tf.keras.layers.Dense(4096, activation = 'relu')(x)
    
    output = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    
    func_model = tf.keras.models.Model(inputs = inputs, outputs=output)
    
    return func_model


# In[5]:


model = create_model()


# compile and fit the data

# In[ ]:


METRICS = [
          'accuracy',
          tf.metrics.TruePositives(name='tp'),
          tf.metrics.FalsePositives(name='fp'),
          tf.metrics.TrueNegatives(name='tn'),
          tf.metrics.FalseNegatives(name='fn'), 
          tf.metrics.Precision(name='precision'),
          tf.metrics.Recall(name='recall'),
          
    ]
adam = tf.keras.optimizers.Adam()    
model.compile(optimizer = adam, loss ='binary_crossentropy', metrics = METRICS)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3
                                                  , patience=5, verbose=2, 
                                                  mode='max')


history = model.fit(train_set, validation_data = val_set, epochs =30, 
                    callbacks=[lr_reduce])


# In[ ]:


model.summary()


# In[ ]:


model.evaluate(test_set)


# plots

# In[ ]:


acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']
#print(history.history['lr'])
epochs=range(len(acc)) # Get number of epochs
rec = history.history['recall']
per = history.history['precision']
val_rec = history.history['val_recall']
val_perc = history.history['val_precision']


# Plot training and validation accuracy per epoch
plt.figure()

plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.legend(['train', 'val'])
plt.title('Training and validation accuracy')


# Plot training and validation loss per epoch
plt.figure()

plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend(['train loss', 'val loss'])
plt.title('Training and validation loss')


#Save Model

model.save('modelv2.h5')

#Load Model

new_model = tf.keras.models.load_model('modelv2.h5')



#GUI

import tkinter as tk   
from keras.preprocessing import image

def write_text():
    print("Tkinter is easy to create GUI!")
    
    
def classify():
    img = image.load_img('./O_98.jpg', target_size=(65, 65))
    x = image.img_to_array(img)/255
    x = np.expand_dims(x, axis=0)
    x.shape
    a=new_model.predict(x)
    if(a[0][0]>0.5):
        print("RECYCLABLE");
    else:
        print("ORGANIC");
    
    
parent = tk.Tk()
frame = tk.Frame(parent)
frame.pack()

text_disp= tk.Button(frame, 
                   text="Hello", 
                   command=classify
                   )

text_disp.pack(side=tk.LEFT)

exit_button = tk.Button(frame,
                   text="Exit",
                   fg="green",
                   command=quit)
exit_button.pack(side=tk.RIGHT)

parent.mainloop()