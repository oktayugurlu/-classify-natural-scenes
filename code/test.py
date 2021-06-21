import cv2
import pandas as pd
from math import log
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt                              #For visualization
from sklearn.preprocessing import normalize
from os import listdir, system
from random import random
from sklearn.metrics import accuracy_score


import sys
SIZE=30
TRAIN_DATA_SIZE=14034
TEST_DATA_SIZE=3650
TEST_DATA_SIZE=3000
train_weight_path=sys.argv[4]
test_path=sys.argv[2]+"/"

#Reading train data
train_image_df = pd.DataFrame(columns=["id", "image", "class name"])

for image_name in listdir(test_path):
    img = cv2.imread(test_path+image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    resized_image = cv2.resize(img, (SIZE, SIZE)) #Resizing image to 30x30
    resized_image /= 255.0   # Normalization
    resized_image = resized_image.reshape(1, SIZE*SIZE) #Vectorizing
    train_image_df = train_image_df.append(pd.DataFrame([[int(image_name.split(".")[0]),resized_image,None]],columns=["id", "image", "class name"]), ignore_index=True)

train_image_df.reset_index(drop=True)
train_image_df.index = train_image_df["id"]
test_label = pd.read_csv('test_label.csv',delimiter=';')
test_label.index = test_label["Id"]
for i in test_label["Id"]:
    train_image_df["class name"][i] = test_label[test_label["Id"]==i]["Category"][i]
            
# Normalize class names as integer value

train_image_df = train_image_df.sample(frac=1,random_state=3)



class Single_Neural_Network:
    def __init__(self, is_sigmoid=False, is_softmax=False, is_tanh=False, is_relu=False, is_mse=False, is_neg_log_likelihood=False ):
        #activation function
        self.is_sigmoid = is_sigmoid
        self.is_softmax = is_softmax
        self.is_tanh = is_tanh
        self.is_relu = is_relu
        
        #objective function
        self.is_mse = is_mse
        self.is_neg_log_likelihood = is_neg_log_likelihood
        
        self.prediction_list=[]
        self.all_epoch_loss_list = []
        
        # weight initialization
        self.weights=np.zeros((6,900))
        for i in range(0,self.weights.shape[0]):
            for j in range(0,self.weights.shape[1]):
                self.weights[i][j] = 0.001
        self.errors=np.zeros((self.weights.shape[0]))
    def __softmax_function(self, output_list): #returns softmax for all output for each train data 
        return np.exp(output_list) / np.sum(np.exp(output_list))
    
    def __der_softmax_function(self, output_list): #returns softmax derivation by input
        gradient=np.zeros((self.weights.shape[0],self.weights.shape[0]))
        for i in range(0,gradient.shape[0]):
            for j in range(0,gradient.shape[0]):
                if i == j:
                    gradient[i][j] = output_list[i] * (1-output_list[i])
                else: 
                    gradient[i][j] = -output_list[i] * output_list[j]
        return gradient
    
    def __forwarding(self, inputs, output_neuron):
        if self.is_sigmoid == True:
            input_for_activation = np.dot(inputs, self.weights[output_neuron])
            return 1/(1+np.exp(-input_for_activation))
        if self.is_softmax == True:
            input_for_softmax_arr = np.dot(inputs, self.weights.T)
            return np.exp(input_for_softmax_arr[output_neuron]) / np.sum(np.exp(input_for_softmax_arr)) # result is (1,output neuron size)
        if self.is_relu == True:   
            input_for_softmax_arr = np.dot(inputs, self.weights.T)
            input_for_softmax_arr[input_for_softmax_arr<=0]=0
            return input_for_softmax_arr[output_neuron]
        else:
            return []    
    def train(self, train_data_df, LEARNING_RATE=0.005, batch_size=1):
        
        # We create weights for each output neuron. Each row indicates weights links to the each output neuron
        output_neuron_size = train_data_df["class name"].unique().shape[0]
        number_of_weights = train_data_df["image"][0].shape[1]
        classes1 = train_data_df["class name"].unique()
        classes2 = train_data_df["class name"].unique()
        classes1.sort()
        classes2.sort()
        
        loss_list = []
        
        #mini batch variables
        new_weights = np.zeros((output_neuron_size, number_of_weights))
        batch_counter=1
        self.batch_size=batch_size
        
        for id_value in train_data_df["id"]: # We update each weights seperately by output neurons.
            
            class_name = train_data_df.loc[id_value]["class name"]
            input_neurons = train_data_df.loc[id_value]["image"][0]
                
            #This gives the true y values of output neurons
            
            expected_output_array = np.zeros((1, output_neuron_size))[0]
            expected_output_array[class_name] = 1
            outputs = np.zeros((self.weights.shape[0]))
            
            #Calculate forward for each output neuron
            if self.is_sigmoid == True:
                for output_neuron in classes1:
                    input_for_activation = np.dot(input_neurons, self.weights[output_neuron])
                    outputs[output_neuron] = 1/(1+np.exp(-input_for_activation))
            if self.is_softmax == True:
                input_for_softmax_arr = np.dot(input_neurons, self.weights.T)
                outputs = self.__softmax_function(input_for_softmax_arr) # result is (1,output neuron size)
            if self.is_relu == True:   
                input_for_softmax_arr = np.dot(input_neurons, self.weights.T)
                input_for_softmax_arr[input_for_softmax_arr<=0]=0
                outputs=input_for_softmax_arr
                
            #Calculate error for each output neuron
            if self.is_mse:
                for output_n in classes2: 
                    self.errors[output_n] += expected_output_array[output_n]-outputs[output_n]

            #mini batch update
            if self.batch_size==batch_counter:
                # backward calculation
                for output_n in classes2:
                    if self.is_sigmoid and self.is_mse:
                        delta=LEARNING_RATE*(self.errors[output_n] / self.batch_size) * outputs[output_n] * (1 - outputs[output_n]) * input_neurons
                    if self.is_softmax and self.is_neg_log_likelihood:
                        delta = LEARNING_RATE * (expected_output_array[output_n]-outputs[output_n]) * input_neurons
                    if self.is_relu and self.is_mse: 
                        delta = LEARNING_RATE * (self.errors[output_n] / self.batch_size) * (0 if outputs[output_n]<0 else 1) * input_neurons
                    self.weights[output_n] += delta
                    
                batch_counter=1
                self.errors=np.zeros((self.weights.shape[0]))
            else:
                batch_counter+=1
                
            if(self.is_mse):
                loss_list.append(np.mean(np.square(expected_output_array-outputs)))
            elif(self.is_neg_log_likelihood):
                loss_list.append(-np.log(outputs[class_name]))

        if(self.is_mse):        
            self.all_epoch_loss_list.append(np.mean(loss_list))
        elif(self.is_neg_log_likelihood):
            self.all_epoch_loss_list.append(np.sum(loss_list))
        
    def predict(self, predict_data):
        self.prediction_list=[]
        output_neuron_size = predict_data["class name"].unique().shape[0]
        for id_value in predict_data["id"]:
            input_neurons = predict_data[predict_data["id"]==id_value]["image"].values[0][0]
            
            outputs = []
            #Calculate forward for each output neuron
            for output_neuron in range(0, output_neuron_size):
                outputs.append(self.__forwarding(input_neurons, output_neuron))
                
            self.prediction_list.append(outputs.index(max(outputs)))
        return self.prediction_list



single_neural_network21 = Single_Neural_Network(is_sigmoid=True, is_mse=True)
single_neural_network21.weights = np.loadtxt(train_weight_path, delimiter=",")
ll=single_neural_network21.predict(train_image_df)
acc=accuracy_score(train_image_df["class name"].values.tolist(), ll)
print("Accuracy: "+str(acc))