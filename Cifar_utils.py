import numpy as np
import os

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



def get_images_from_files (CIFAR_DIR = 'cifar-10-batches-py'):
    dirs = ['batches.meta','data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
    all_data = [0,1,2,3,4,5,6]
    
    print("Unpickle the data")
    
    for i,direc in zip(all_data,dirs):
        all_data[i] = unpickle(os.path.join(CIFAR_DIR,direc))
        
    batch_meta = all_data[0]
    data_batch1 = all_data[1]
    data_batch2 = all_data[2]
    data_batch3 = all_data[3]
    data_batch4 = all_data[4]
    data_batch5 = all_data[5]
    test_batch = all_data[6]
    
    print("Setting Up Training Images and Labels")
    
    all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
    training_images = np.vstack([d[b"data"] for d in all_train_batches])
    train_len = len(training_images)
    trainX = training_images.reshape(train_len,3,32,32).transpose(0,2,3,1)
    
    trainY = np.hstack([d[b"labels"] for d in all_train_batches])
    #trainY.shape
    
    print("Setting Up Test Images and Labels")
    
    test_images = np.vstack([d[b"data"] for d in [test_batch]])
    test_len = len(test_images)
    testX = test_images.reshape(test_len,3,32,32).transpose(0,2,3,1)
    #testX.shape
    
    testY = np.hstack([d[b"labels"] for d in [test_batch]])
    
    print("Dataset loaded successfully")
    
    return ((trainX, trainY), (testX, testY))
    