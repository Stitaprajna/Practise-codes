#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow-gpu')


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


get_ipython().system('nvidia-smi')


# In[2]:


# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt


# In[20]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'C:/Users/Dinesh_PC/STOCK_DL/Test'
valid_path = 'C:/Users/Dinesh_PC/STOCK_DL/Train'


# In[21]:



inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[22]:


# don't train existing weights
for layer in inception.layers:
    layer.trainable = False


# In[23]:


# useful for getting number of output classes
folders = glob('C:/Users/Dinesh_PC/STOCK_DL/Train')


# In[24]:


folders


# In[25]:


# our layers - you can add more if you want
x = Flatten()(inception.output)


# In[45]:


prediction = Dense(3, activation='softmax')(x)


# create a model object
model = Model(inputs=inception.input, outputs=prediction)


# In[46]:


model.summary()


# In[47]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[48]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[49]:


# Make sure you provide the same target size as initialied for the image sizen
training_set = train_datagen.flow_from_directory('C:/Users/Dinesh_PC/STOCK_DL/Train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[50]:


test_set = test_datagen.flow_from_directory('C:/Users/Dinesh_PC/STOCK_DL/Test',
                                            target_size = (224, 224),
                                            batch_size = 16,
                                            class_mode = 'categorical')


# In[52]:



# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=100,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[53]:


import matplotlib.pyplot as plt


# In[54]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[56]:



from tensorflow.keras.models import load_model

model.save('model_incept.h5')


# In[57]:


y_pred = model.predict(test_set)


# In[58]:


y_pred


# In[59]:


import numpy as np
y_pred = np.argmax(y_pred, axis=1)


# In[60]:


y_pred


# In[73]:


from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf


# In[74]:


model=load_model('model_incept.h5')


# In[83]:


img=image.load_img('D:/Dileep/Stock_market/DL_Stock/Stock_Analysis/Final_SS/prince.jpg',target_size=(224,224))


# In[84]:


x=image.img_to_array(img)
x


# In[85]:


x.shape


# In[86]:


x=x/255


# In[87]:


import numpy as np
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape


# In[88]:


model.predict(img_data)


# In[ ]:





# In[ ]:




