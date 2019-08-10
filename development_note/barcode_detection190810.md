

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
import pandas as pd
import os
from keras.preprocessing.image import ImageDataGenerator
```

    Using TensorFlow backend.
    


```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```


```python
from keras.optimizers import Adam
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model
```


```python
from keras import layers
```


```python
from keras import models
```


```python
train_datagen = ImageDataGenerator(rescale= 1./255)
```


```python
validation_datagen = ImageDataGenerator(rescale= 1./255)
```

인풋 이미지 사이즈에 따라 해상도가 변화한다


```python
image_input_size = 122
```


```python
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\A018025\Desktop\cat_vs_barcode_train',
    target_size = (image_input_size,image_input_size),
    batch_size = 16,
    class_mode='binary')
```

    Found 2239 images belonging to 2 classes.
    

변환된 이미지를 보고 싶다면 해당 명령어를 실행

plt.imshow(train_generator[0][0][5])
plt.show()


```python
validation_generator = validation_datagen.flow_from_directory(
    r'C:\Users\A018025\Desktop\cat_vs_barcode_validation',
    target_size = (image_input_size,image_input_size),
    batch_size = 16,
    class_mode='binary')
```

    Found 220 images belonging to 2 classes.
    

Mobilenet 모델 개발


```python
def build_model_mobile(image_input_size):
    target_size=image_input_size
    input_tensor = Input(shape=(target_size, target_size, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(target_size, target_size, 3),
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers
        
    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)
    
    ##
    # softmax: calculates a probability for every possible class.
    #
    # activation='softmax': return the highest probability;
    # for example, if 'Coat' is the highest probability then the result would be 
    # something like [0,0,0,0,1,0,0,0,0,0] with 1 in index 5 indicate 'Coat' in our case.
    ##
    output_tensor = Dense(1,activation='sigmoid')(op)
    #softmax 를 sigmoid로 바꾸니  acc가 0.45 -> 0.95로 상승했따.

    model = Model(inputs=input_tensor, outputs=output_tensor)


    return model
```

 커스텀 모델 개발


```python
def custom_model(image_input_size):
    target_size=image_input_size    
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3, 3), activation='relu', input_shape=(image_input_size,image_input_size,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3), activation= 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3), activation= 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3), activation= 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

```


```python
model = build_model_mobile(image_input_size)
```


```python
model = custom_model(image_input_size)
```

    WARNING:tensorflow:From C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    


```python
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 120, 120, 32)      896       
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 60, 60, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 58, 58, 64)        18496     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 29, 29, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 27, 27, 128)       73856     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 13, 13, 128)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 11, 11, 128)       147584    
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 5, 5, 128)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 3200)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 512)               1638912   
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 1,880,257
    Trainable params: 1,880,257
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=['acc'])
```


```python
model.fit_generator(generator = train_generator, steps_per_epoch = 100, verbose =1 , epochs= 3,validation_data=validation_generator, validation_steps=50 )
```

    WARNING:tensorflow:From C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/3
     88/100 [=========================>....] - ETA: 5s - loss: 0.3275 - acc: 0.8338

    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 80000 bytes but only got 0. Skipping tag 64640
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 65536 bytes but only got 0. Skipping tag 3
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 404094976 bytes but only got 0. Skipping tag 5
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 404619264 bytes but only got 0. Skipping tag 5
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 131072 bytes but only got 0. Skipping tag 3
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 425459712 bytes but only got 0. Skipping tag 4
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 1385474 bytes but only got 6833. Skipping tag 513
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 3846701056 bytes but only got 0. Skipping tag 2
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 3300917248 bytes but only got 0. Skipping tag 7
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:754: UserWarning: Possibly corrupt EXIF data.  Expecting to read 196867 bytes but only got 6833. Skipping tag 0
      " Skipping tag %s" % (size, len(data), tag))
    C:\Users\A018025\AppData\Local\Continuum\anaconda3\lib\site-packages\PIL\TiffImagePlugin.py:771: UserWarning: Corrupt EXIF data.  Expecting to read 12 bytes but only got 8. 
      warnings.warn(str(msg))
    

    100/100 [==============================] - 55s 550ms/step - loss: 0.3088 - acc: 0.8469 - val_loss: 0.1013 - val_acc: 0.9480
    Epoch 2/3
    100/100 [==============================] - 50s 497ms/step - loss: 0.0869 - acc: 0.9725 - val_loss: 0.0762 - val_acc: 0.9617
    Epoch 3/3
    100/100 [==============================] - 49s 487ms/step - loss: 0.0721 - acc: 0.9725 - val_loss: 0.0180 - val_acc: 0.9924
    




    <keras.callbacks.History at 0x2460593d518>




```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 17930865965020647171
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 4943917875
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 17849595507759766617
    physical_device_desc: "device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:01:00.0, compute capability: 6.1"
    ]
    
