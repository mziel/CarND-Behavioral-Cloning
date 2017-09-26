
## Behavioral Cloning Project

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

### Rubric Points

**Here I will consider the rubric points individually and describe how I addressed each point in my implementation.**

### Files Submitted & Code Quality

#### 1. Are all required files submitted?

My project includes the following files:

* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `CarND-Behavioral-Cloning-Writeup.md` and `CarND-Behavioral-Cloning-Writeup.ipynb` summarizing the results (you're reading it now!)
* `video.mp4` with the recorded track behavior

#### 2. Is the code functional?

You can use `python drive.py model.h5` to steer the car

#### 3. Is the code usable and readable?

The `model.py` is the main entry point to the project:

* function `load_lines` reads the lines from the simulator output (driving log)
* function `generator` uses Python generator to generate input samples for training and validation, because the amount of data is too large to fit in memory
    * it uses `translate_image` and `flip_image` explained later
* function `load_generator` calls `load_lines` and `generator` to prepare the data for trainign
* function `steering_model` defines a keras model I have used. It also optionally prints the network structure.
* function `fit_model` calls the `steering_model` and `load_generator` to perform `fit_generator` operation and fit the neural network using Adam optimizer
* function `visualize_model` the training history

### Model Architecture and Training Strategy

#### 1. Has an appropriate model architecture been employed for the task?

The starting point for the model was the architecture described in `End to End Learning for Self-Driving Cars`. Apart from the changed below I have also adjusted parameters of this architecture (layer sizes and depths) as an iterative process during training.

* The model contains convolutional, pooling and dense (with ReLu activation) trainable layers.
* I have added rescaling Keras Labmda layer to normalize the input with GPU parallelization
* I have added Cropping2D to focus the network on the road and reduce number of parameters by 2x
* I have added Dropout layers to combat overfitting

Below you can see the full description of the model:


```python
from model import steering_model
steering_model(True)
```

    Using TensorFlow backend.


    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lambda_1 (Lambda)            (None, 160, 320, 3)       0         
    _________________________________________________________________
    cropping2d_1 (Cropping2D)    (None, 90, 320, 3)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 88, 318, 64)       1792      
    _________________________________________________________________
    activation_1 (Activation)    (None, 88, 318, 64)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 86, 316, 32)       18464     
    _________________________________________________________________
    activation_2 (Activation)    (None, 86, 316, 32)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 84, 314, 16)       4624      
    _________________________________________________________________
    activation_3 (Activation)    (None, 84, 314, 16)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 82, 312, 8)        1160      
    _________________________________________________________________
    activation_4 (Activation)    (None, 82, 312, 8)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 41, 156, 8)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 41, 156, 8)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 51168)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 16)                818704    
    _________________________________________________________________
    activation_5 (Activation)    (None, 16)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 16)                272       
    _________________________________________________________________
    activation_6 (Activation)    (None, 16)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 16)                272       
    _________________________________________________________________
    activation_7 (Activation)    (None, 16)                0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 16)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 845,305
    Trainable params: 845,305
    Non-trainable params: 0
    _________________________________________________________________





    <keras.models.Sequential at 0x7f18086a47f0>



#### 2. Has an attempt been made to reduce overfitting of the model?

I have employed following strategies to reduce overfitting:
* The model contains dropout layers
* The model was trained and validated on different data sets
* The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track
* Dataset was augmented with small affine translations (`translate_image`), mirror images (`flip_image`) and multiple camera adjustments to minimize the reliance on the track

#### 3. Have the model parameters been tuned appropriately?

The model used an adam optimizer, so the learning rate was not tuned manually.

The parameter I have experimented with is the multiple camera correction. `0.25` gave me visibly better results than `0.2`.

#### 4. Is the training data chosen appropriately?

Training data was chosen to keep the vehicle driving on the road. I have used the dataset provided in the lecture resources, but I augmented it in the following ways.

* Usage of multiple cameras with 0.25 correction to ensure recovery from deviations.
* Horizontal flipping of images to combat the steer-left nature of the track. (`flip_image`). Every sample can be flipped or not with probability 0.5.
* Random small affine transformations help generalize the training input (`translate_image`). Every time I draw a sample from a generator I translate the image a little and adjust the angle to make up for it.

### Architecture and Training Documentation

#### 1. Is the solution design documented?

The overall strategy for deriving a model architecture was to make sure the model generalizes the image input (hence the usage of convolutional and maxpooling layers).

The starting point for the model was the architecture described in `End to End Learning for Self-Driving Cars`. Although I have changed the architecture somewhat most of the improvements came from the data augmentation. The design choices have been described in the cells above.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Is the model architecture documented?

In the cell above you can see the model architecture.

#### 3. Is the creation of the training dataset and training process documented?


```python
from model import load_lines
```


```python
lines = load_lines()
len(lines)
```

    ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']





    8036



I have used the dataset provided in the lecture resources. It contained `8036` intial samples. I have augmented those with the use of multiple cameras (3x) and image flipping (2x), which yielded `8036 * 2 * 3 = 48 216` potential samples. I have used a generator with batch 16 to randomly sample those images. 

As described above I have also used random translations on every image to make the model generalize outside the training situation.

The data points have been randomly shuffled the data set and 20$ of the data was put into a validation set.

This amount of data resulted in memory issues so I needed to use generators to feed the batches to the model.

### Simulation

#### 1. Is the car able to navigate correctly on test data?

I enclose `video.py` file to show how the model performed.


```python
from IPython.display import HTML

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format("video.mp4"))
```





<video width="960" height="540" controls>
  <source src="video.mp4">
</video>



