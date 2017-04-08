
# coding: utf-8

# In[1]:

'''

Start with VGG16 network, not including last fully connected layers.

Train a simple fully connected layer on top of these.


'''

import numpy as np
import keras
import random
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
import vgg16 as VGG
import prepare.collect as pc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
set_session(tf.Session(config=config))
'''
N_CLASSES = 2
IMSIZE = (224, 224)

XML_DIR = "../data/annotations/xmls/"
IMG_DIR = "../data/images/"
VAL_RATIO = 0.3

# Start with an VGG16 model, not including the final dense layers.
base_model = VGG.VGG16(weights='imagenet')
print ('Loaded vgg16 model')


# In[2]:

# Turn off training on base model layers
for layer in base_model.layers:
    layer.trainable = False
    
base_model_last = base_model.get_layer('flatten').output
x = Dense(4096, activation='relu', name='fc1-1')(base_model_last)
x = Dense(4096, activation='relu', name='fc1-2')(x)
x = Dropout(0.5,  name='dp1-1')(x)
predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

y = Dense(4096, activation='relu', name='fc2-1')(base_model_last)
y = Dense(4096, activation='linear', name='fc2-2')(y)
y = Dropout(0.5,  name='dp2-1')(y)
aux_predictions = Dense(4, activation='linear', name='aux_predictions')(y)

#model = Model(input=base_model.input, output=predictions)
#model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])


model = Model(input=base_model.input, output=[predictions, aux_predictions])

#not training bounding box part
model.get_layer("fc1-1").trainable=False
model.get_layer("fc1-2").trainable=False
model.get_layer("predictions").trainable=False

#model.get_layer("block5_conv2").trainable=True;
#adam = optimizers.Adam(lr=0.00001)
sgd = optimizers.SGD(lr=0.00001)
model.compile(optimizer=sgd,
              loss={'predictions': 'categorical_crossentropy', 'aux_predictions': 'mean_squared_error'},
             loss_weights={'predictions': 0, 'aux_predictions': 1}, metrics=['accuracy'])


# In[3]:

# Show some debug output
print (model.summary())


# In[4]:

#for oxford dataset
xmlFiles = pc.listAllFiles(XML_DIR)
infoList = list(map(lambda f:pc.getInfoTupleForXml(f,IMG_DIR) ,xmlFiles))

random.shuffle(infoList)
cutIndex = int(len(infoList)*VAL_RATIO)
train_files_ox = infoList[:cutIndex]
val_files_ox = infoList[cutIndex:]
#------------------


# In[5]:

# for kaggle dataset
KAGGLE_TRAIN_DIR="../../Mind/A3/data/catdog/train/"
KAGGLE_VAL_DIR="../../Mind/A3/data/catdog/validation/"
def processKaggleFolder(imgDir, label):
    files = pc.listAllFiles(imgDir)
    files = list(map(lambda f:[0,f,label,(50,50,50,50)],files))
    return files
catfiles_train=processKaggleFolder (KAGGLE_TRAIN_DIR+"cat/",0)
dogfiles_train =processKaggleFolder (KAGGLE_TRAIN_DIR+"dog/",1)
train_files_kaggle = catfiles_train + dogfiles_train
catfiles_val = processKaggleFolder (KAGGLE_VAL_DIR+"cat/",0)
dogfiles_val = processKaggleFolder (KAGGLE_VAL_DIR+"dog/",1)
val_files_kaggle = catfiles_val+dogfiles_val
train_files = train_files_ox # + train_files_kaggle
val_files = val_files_ox # + val_files_kaggle
np.random.seed()
random.shuffle(train_files)
random.shuffle(val_files)


# In[6]:


#print(val_files)
#np.random.seed()
img_datagen = ImageDataGenerator(rescale=1./255)

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def my_load_img(img_path,img_datagen,size):
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = img_datagen.random_transform(x)
    x = img_datagen.standardize(x)
    return x
def my_img_generator(files,img_datagen,batch_size):
    
    index = 0
    count = 0
    img_datas=[]
    img_labels=[]
    img_bboxes=[]
    while 1:
        if count < batch_size:
            img_datas.append(my_load_img(files[index][1],img_datagen,IMSIZE))
            img_labels.append(files[index][2])
            
            img_bboxes.append(np.array(files[index][3]))
            
            index=(index+1)%len(files)
            count+=1
        else:
            count=0
            one_hot_labels=to_categorical(img_labels, num_classes=2)
            yield (np.array(img_datas),[np.array(one_hot_labels),np.array(img_bboxes)])
            img_datas = []
            img_labels = []
            img_bboxes=[]

batch_size=32

my_train_generator = my_img_generator(train_files,img_datagen,batch_size)
my_val_generator = my_img_generator(val_files,img_datagen,batch_size)


# In[7]:

model.load_weights('catdog_combine_3.h5',by_name=True) 


# In[8]:

model.fit_generator(
        my_train_generator,
        samples_per_epoch=128,
        nb_epoch=1,
        validation_data=my_val_generator,
        verbose=2,
        nb_val_samples=128)
#model.save_weights('catdog_combine_try.h5')


# In[9]:

def getPredForImg(img_path):
    x = my_load_img(img_path,img_datagen,IMSIZE)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds
def testImage(img_path,preds):  
    size = Image.open(img_path).size
    width=size[0]
    height=size[1]
    im = np.array(Image.open(img_path), dtype=np.uint8)
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    actual_preds=[preds[0]*width*0.01,preds[1]*height*0.01,preds[2]*width*0.01,preds[3]*height*0.01]
    # Create a Rectangle patch
    rect = patches.Rectangle((actual_preds[0],actual_preds[1]),actual_preds[2]-actual_preds[0],actual_preds[3]-actual_preds[1],linewidth=1,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()
    return
files = pc.listAllFiles("../data/test/")
random.shuffle(files)


# In[21]:


for f in files[60:70]:
    preds = getPredForImg(f)
    print(preds[0][0])
    print('cat' if preds[0][0][0]>0.5 else 'dog')
    testImage(f,preds[1][0])



# In[22]:


def showTest(img_dir):
    preds = getPredForImg(img_dir)
    print(preds[0][0])
    #print('cat' if preds[0][0][0]>0.5 else 'dog')
    testImage(img_dir,preds[1][0])
showTest("cat.jpg")
showTest("../data/elephant.jpg")
showTest("../data/bill.jpg")
showTest("../data/li.jpg")

