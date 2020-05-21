
# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# In[3]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels ) ,(test_images, test_labels ) = fashion_mnist.load_data()


# In[4]:


train_images.shape


# In[5]:


type(train_images)


# In[6]:


train_images[0,23,23]


# In[16]:


train_labels[:10]


# In[8]:


class_names = [ 'T-shirt/top' , 'Trouser' , 'Dress', 'Coat' , 'Sandal' , 'Sneaker', 'Bag', 'Ankle boot']


# In[13]:


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# ## data preprocessing
# train_images = train_images / 255.0
# 
# test_images = test_images / 255.0

# In[35]:


#creating the model


model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),        #input layer (flattens into a 1D array)
    keras.layers.Dense(128, activation= 'relu'),     #hidden layer  (leniar unit  )
    keras.layers.Dense(120,activation = 'softmax')     #output layer (0 / 1)
    ])


# In[36]:


#compiling the model 

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[38]:


#TRAINING THE MODEL
model.fit(train_images, train_labels , epochs =10)


# In[39]:


#EVALUATING THE MODEL
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test Accuracy is:',test_acc)


# In[ ]:


#MAKING PREDICTIONS
predictions =model.predict(test_images)
print(class_names[np.argmax(predictions[5])])
plt.figure()
plt.imshow(test_images[5])      #check for any number in the range of class_names
plt.colorbar()
plt.grid(False)
plt.show()


