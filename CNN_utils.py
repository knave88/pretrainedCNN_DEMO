import numpy as np
from scipy.ndimage import zoom

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from keras import backend as K
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.applications.mobilenet import preprocess_input, decode_predictions


def getLayerIndexByName(model, layername):
    for idx, layer in enumerate(model.layers):
        if layer.name == layername:
            return idx
        
def get_heatmap(model, image, width_factor, height_factor):
    conv_out, pred = model.predict(image)
    conv_out = np.squeeze(conv_out)
    pred = np.argmax(pred)
    
    mat_for_mult = zoom(conv_out, (224./width_factor, 224./height_factor, 1), order=1)
    weights = model.layers[-3].get_weights()[0][:,:,:,int(pred)]
    weights = np.squeeze(weights)
    out = np.dot(mat_for_mult.reshape((224*224, 1024)), weights).reshape((224, 224))
    #out = 255-out
    return pred, out

def get_bounds(out, percentile=95):
    # Get bounding box of 95+ percentile pixels
    a = out.flatten()
    filtered = np.array([1 if x > np.percentile(a, percentile) else 0 for x in a]).reshape(224,224)
    left, up, down, right = 224, 224, 0, 0
    for x in range(224):
        for y in range(224):
            if filtered[y,x] == 1:
                left = min(left, x)
                right = max(right, x)
                up = min(up, y)
                down = max(down, y)
    return left, up, down, right


def heatmap_for_top_pred (img_path, model, figsizeX):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    conv_out, preds = model.predict(x)
    decode_preds = decode_predictions(preds, top=13)
    
    analysed_preds = 0
    top13_preds = preds[0].argsort()[-13:][::-1]
    analyzed_class = top13_preds[analysed_preds]

    top_class_output = model.output[1][:, analyzed_class]
    last_conv_layer = model.get_layer('conv_pw_13')

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
        
    heatmap = cv2.resize(heatmap, (224,224))
    heatmap = np.uint8(255 * heatmap)
    
    left, up, down, right = get_bounds(heatmap, percentile=95)

    rect = patches.Rectangle((left, up), (right-left), (down-up), linewidth=1,  edgecolor='r', facecolor='none')
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(figsizeX, figsizeX))
    axes.imshow(img, alpha=0.7)
    axes.imshow(heatmap, cmap='jet', alpha=0.3)
    left, up, down, right = get_bounds(heatmap, percentile=95)
    rect = patches.Rectangle((left, up), (right-left), (down-up), linewidth=1,  edgecolor='r', facecolor='none')
    axes.add_patch(rect)
    axes.set_title('Heat map and bounding box for prediction: '+str(decode_preds[0][analysed_preds][1]))
    
    return None

def print_learning_acc(myhist):
    # summarize history for accuracy
    plt.plot(myhist['accuracy'])
    plt.plot(myhist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(myhist['loss'])
    plt.plot(myhist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    return None

