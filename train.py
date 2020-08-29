'''
  Author       : Bao Jiarong
  Creation Date: 2020-08-28
  email        : bao.salirong@gmail.com
  Task         : Autoencoder Implementation
  Dataset      : cityscapes
'''
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import random
import cv2
import loader
import unet

np.random.seed(7)
tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
width      = 572 #>> 1  # = 286
height     = 572 #>> 1  # = 286
channel    = 3
model_name = "models/cityscapes_2/checkpoint"
data_path  = "../../data_img/cityscapes_data/train/"

# Step 0: Global Parameters
epochs     = 100
lr_rate    = 0.00001
batch_size = 4

# Step 1: Create Model
model = unet.UNet((height, width, channel),units=64)

if sys.argv[1] == "train":

    print(model.summary())
    # sys.exit()

    # Load weights:
    try:
        model.load_weights(model_name)
    except:
        print("no weights were found")

    # Step 3: Load data
    # X_train, Y_train, X_valid, Y_valid = loader.load_light(data_path,width,height,True,0.8,False)
    X_train, Y_train, X_valid, Y_valid = loader.load_heavy(data_path,width,height,False,0.8,False)

    # Define The Optimizer
    #---------------------
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_rate)

    # Define The Loss
    #---------------------
    @tf.function
    def my_loss(y_true, y_pred):
        return tf.keras.losses.MSE(y_true=y_true, y_pred=y_pred)

    # Define The Metrics
    #---------------------
    tr_loss = tf.keras.metrics.MeanSquaredError(name = 'tr_loss')
    va_loss = tf.keras.metrics.MeanSquaredError(name = 'va_loss')

    #---------------------
    @tf.function
    def train_step(X, Y_true):
        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)
            loss   = my_loss(y_true=Y_true, y_pred=Y_pred )
        gradients= tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        tr_loss.update_state(y_true = Y_true, y_pred = Y_pred )

    #---------------------
    @tf.function
    def valid_step(X, Y_true):
        Y_pred= model(X, training=False)
        loss  = my_loss(y_true=Y_true, y_pred=Y_pred)
        va_loss.update_state(y_true = Y_true, y_pred = Y_pred)

    #---------------------
    # start training
    L = len(X_train)
    M = len(X_valid)
    steps  = int(L/batch_size)
    steps1 = int(M/batch_size)

    for epoch in range(epochs):
        # Run on training data + Update weights
        for step in range(steps):
            # images, labels = loader.get_batch_light(X_train, Y_train, batch_size, width, height)
            images, labels = loader.get_batch_heavy(X_train, Y_train, batch_size)
            train_step(images,labels)

            print(epoch,"/",epochs,step,steps,"loss:",tr_loss.result().numpy(),end="\r")

        # Run on validation data without updating weights
        for step in range(steps1):
            # images, labels = loader.get_batch_light(X_valid, Y_valid, batch_size, width, height)
            images, labels = loader.get_batch_heavy(X_valid, Y_valid, batch_size)
            valid_step(images, labels)

        print(epoch,"/",epochs,
              "loss:",tr_loss.result().numpy(),
              "val_loss:",va_loss.result().numpy())

        # Save the model for each epoch
        model.save_weights(filepath=model_name, save_format='tf')

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    img_name = "../../data_img/cityscapes_data/val/10.jpg"
    img = cv2.imread(img_name)
    w = img.shape[1] >> 1
    img  = img[:,:w,:]
    image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    images = np.array([image])/255.0
    # images = loader.scaling_tech(images,method="normalization")

    # Step 5: Predict the class
    preds = my_model.predict(images)
    preds = (preds[0] - preds.min())/(preds.max() - preds.min())
    # preds = np.dstack((preds,preds,preds))
    print(preds.shape)
    print(images.shape)
    images = np.hstack((images[0],preds))
    # images = cv2.resize(images,(width*4,height*2))
    cv2.imshow("imgs",images)
    cv2.waitKey(0)
elif sys.argv[1] == "predict_all":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    imgs_filenames = sorted([os.path.join("../../data_img/cityscapes_data/val/", file)
                             for file in os.listdir("../../data_img/cityscapes_data/val/")],
                             key=os.path.getctime)[2:5]
    images = []
    for filename in imgs_filenames:
        img = cv2.imread(filename)
        w = img.shape[1] >> 1
        img  = img[:,:w,:]
        image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
        images.append(image)

    # True images
    images = np.array(images)
    images = loader.scaling_tech(images,method="normalization")

    # Predicted images
    preds = my_model.predict(images)
    preds = (preds - preds.min())/(preds.max() - preds.min())


    true_images = np.hstack(images)
    pred_images = np.hstack(preds)

    images = np.vstack((true_images, pred_images))
    h = images.shape[0]
    w = images.shape[1]
    images = cv2.resize(images,(int(w * 0.7), int(h * 0.7)))

    cv2.imshow("imgs",images)
    cv2.waitKey(0)
elif sys.argv[1] == "video":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    cap = cv2.VideoCapture(sys.argv[2])

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        img = frame

        image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
        images = np.array([image])/255.0
        # images = loader.scaling_tech(images,method="normalization")

        # Step 5: Predict the class
        preds = my_model.predict(images)
        preds = (preds[0] - preds.min())/(preds.max() - preds.min())
        # preds = np.dstack((preds,preds,preds))
        # print(preds.shape)
        # print(images.shape)
        images = np.hstack((images[0],preds))
        # images = cv2.resize(images,(width*4,height*2))

        # Display the resulting frame
        cv2.imshow('frame',images)
        if cv2.waitKey(1) == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
