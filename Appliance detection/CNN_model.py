import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import *
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


data = pd.read_csv("data.csv")
#print(data)

n = int(data.max(axis = 0, skipna = True)[1]) + 1  # gets the number of readings
h = int(data.max(axis = 0, skipna = True)[0]) # gets the number of houses

houses = np.zeros((n,5,h))  # time, tv inst power, aggr power, filtered tv , on/off

for i in range(h):
    houses[:,0:3,i] =  data[data['House']==i+1  ].values[:,1:]   # data is now seperated by house

# visulise data
plt.figure()
plt.suptitle('Tv power and agg power seperated ')
plt.subplot(211)
for i in range(h):
    plt.plot(houses[:,0,i],houses[:,1,i], label = 'House ' + str(i+1) )
plt.subplot(212)
for i in range(h):
    plt.plot(houses[:,0,i],houses[:,2,i], label = 'House ' + str(i+1) )
plt.legend()
# plt.show()

plt.figure()
plt.suptitle('Normalised data, shown by house')
for i in range(h):
    plt.subplot(3,1,i+1)
    plt.plot(houses[:,0,i],houses[:,1,i]/np.average(houses[:,1,i]) )
    plt.plot(houses[:, 0, i], houses[:, 2, i]/np.average( houses[:, 2, i]) )
plt.legend()
plt.show()


#house one has a standbymode, and low power usage when on, very noisy
#house two is on or off , sharp pulses when on, large power
# house three has high variation in power when off flicks between 0-20 , when on still very noisy and non constant power usage

#model has to
#account for variation in power when on
#account for a standby mode
#account for the different power shapes


#    TASK 1

#filter and normalise the Tv instantuos power
houses[:,3,:] = savgol_filter(houses[:, 1, :] / np.average(houses[:, 1, :],axis=0), 11, 2,axis=0)
thres = 1.15  # the threshold for seperation
plt.figure()
plt.suptitle('Filtered inst power, and determined state')
for i in range(h):
    houses[ np.argwhere( houses[:,3,i] >= thres ) ,4,i  ] +=1   # makes the 4th row qual to 1 if the filtered result is higher then the threshold
    plt.subplot(3, 1, i + 1)
    plt.plot(  houses[:, 0, i],houses[:, 3, i]  ) # plot filtered curve
    plt.plot(houses[:, 0, i], houses[:, 4, i]) # plot on state
    plt.plot(houses[:, 0, i], houses[:, 1, i] / np.average(houses[:, 1, i])) # plot normalised curve
plt.show()


# plt.figure()
# for i in range(h):
#     plt.subplot(3, 1, i + 1)
#     d  = np.gradient(houses[:, 2, i] / np.average(houses[:, 2, i]))
#     plt.plot(houses[:, 0, i], houses[:, 4, i])  # plot on state
#     plt.plot(houses[:, 0, i], d) # plot normalised curve
# plt.show()


# TASK 2

#take aggregated power as an input
#can use previous task as labels in supervised learning

#distinktive patterns in Tv power form
# - sharp increase/decrease when turning off/on
# - increase of 20, 80 or 130 so a variable increase
# - correlation between time, when its on its likely the next timestep is also on

#problems in aggregated power
# - very large peaks present
# - some periodic peaks present from other appliances, these are at different frequencies in each house
# - noise/ peak flatness  is of similar order to tv power usage in house one

# thoughts
# - test/check for against on/off states
# - test/check for times where state changes
# - use a CNN/ANN/MLP for classification
#   - should the input then be the whole time, or chucks that are N timesteps big
#   - N timesteps big means it can be generalised to different test input sizes
#   - should then have the training set and test set build up of these N big inputs, with each one slide along
#   - allows for genralisation in position of on state within input
#   - output N 1/0 (or probability of being on, and then threshold) outputs corresponding to each timestep
#   -


# online readings, for ideas about usual form of networks, these were usually on predicition occurances of a certain appliance and not on exact time
# http://aircconline.com/ijaia/V9N2/9218ijaia06.pdf
#


def sep_data(houses, T=0.8):
    # seperate the house data into windowed sections winin each house
    # split into training (T%) and test set, no validation set is being used
    # record the class as a one hot vector of if the last state is on/off

    window_len = 20
    #at size 20 the classifier was classfiing the periodic sections as the tv being on in house one #filter sized 6,3 wights 1.8,1
    #at length 30 a similar thing happened, filters 6,3 weights at 3:1, not as accurate as ^
    # at 10, with 3,3, stride and 2.2:1 similar problems

    ntrain = int((n-window_len)*T) # amount of data from each house to train on
    train = np.zeros((ntrain*h,window_len,1,2))                         # [#windows x samples per window x 1(as only one input feature) x 2 (for rata, and weights)
    test = np.zeros( ( (n-window_len-ntrain)*h,window_len,1,2)  )
    for j in range(h): #each house
        for i in range(n-window_len):
            train_index = i+j*ntrain ## which row in the reformatted array is being filled
            test_index = i-ntrain + j*(n-window_len-ntrain)

            if i<ntrain:  #part of training set
                train[train_index,:,:,0 ] =np.reshape(houses[i:i+window_len,2,j] , (window_len,1))     # window up the aggregated power
                #train[train_index, :,:,1] = np.reshape(houses[i:i + window_len,4, j], (window_len,1)) # no longer used, was used when every state in window was a class
                train[train_index, 0 , :, 1] = houses[i + window_len, 4, j]                            # identify state of last step in window
                train[train_index, 1, :, 1] = -(train[train_index, 0 , :, 1] -1)                       # one hot encode the category,cat 0 = on, cat 1 = off
            else: # test set
                test[test_index , :,:, 0] =np.reshape( houses[i:i + window_len, 2, j], (window_len,1))
                #test[test_index), :,:, 1] = np.reshape(houses[i:i + window_len, 4, j], (window_len,1))
                test[test_index, 0, :, 1] = houses[i + window_len, 4, j]
                test[test_index, 1, :, 1] = -(test[test_index, 0, :, 1] - 1)
    #return train[:,:,:,0],train[:,:,0,1],test[:,:,:,0],test[:,:,0,1]

    # how uneven is the data? this could be a problem when training, as a large group can overpower and always be predicted
    wratio = np.sum(train[:,1,:,1])/np.sum(train[:,0,:,1])
    print(wratio)
    return train[:,:,:,0],train[:,0:2,0,1],test[:,:,:,0],test[:,0:2,0,1]



# fit a model to the training set and evaluate on test set
def run_model(trainx, trainy, testx, testy):
    verbose, epochs, batch_size = 0, 10, 64
    n_timesteps, n_features, n_outputs = trainx.shape[1], trainx.shape[2], trainy.shape[1]

    # Build a convolutional neural net using keras
    # the convolution should hopefully identify features in the windowed time series dat, to indicate if the final times state
    #
    model = Sequential()
    model.add(Conv1D(filters=120, kernel_size=6, activation='sigmoid', input_shape=(n_timesteps, n_features)))
    model.add(Conv1D(filters=120, kernel_size=3, activation='sigmoid'))
    model.add(Dropout(0.5)) #this layer helps regulasrization, and hopefully reduces overfitting to the training set
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=100, kernel_size=3, activation='relu'))
    model.add(Flatten()) # flattens the two dimension array to a 1d vector
    model.add(Dense(100, activation='sigmoid')) #fully connected layer
    model.add(Dense(n_outputs, activation='softmax'))  # use softmax here to create a probabilty of residing in each class

    #used when each timestep was being predicted as either on or off, sigmoid function as multiple classes are possible, and different loss metric reuired
    #model.add(Dense(n_outputs, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.summary()
    model.fit(trainx, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, class_weight = {0:2.2,1:1 }) # class weights are implemented to fix the class imbalance, and stop off being predicted for all states as often

    # evaluate model
    pr = model.predict(testx)
    pt = model.predict(trainx)

    plt.figure()
    ax1 = plt.subplot(211)
    ax1.title.set_text('Training set')
    ax1.plot(np.round(pt[:,0]), label = 'Predicted class')
    ax1.plot(trainy[:,0], label ='class (1=on) ' )
    ax1.plot(pt[:, 0], label='Class probability')


    ax2 = plt.subplot(212)
    ax2.title.set_text('Test set')
    ax2.plot(np.round(pr[:, 0]), label = 'Predicted class')
    ax2.plot(testy[:, 0],label ='class (1=on) ' )
    ax2.plot(pr[:, 0],label='Class probability')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3)

    plt.show()  #plot the predictions

    loss1, accuracy = model.evaluate(testx, testy, batch_size=batch_size, verbose=0)
    loss2, at = model.evaluate(trainx, trainy, batch_size=batch_size, verbose=0)
    ## 65% accuracy is roughly expected if the classifier says the tv is always off, as this class is larger
    print('Training accuracy: %.3f ' % at)
    print('Training loss: %.3f ' % loss2)
    print('Test accuracy: %.3f' % accuracy)
    print('Test loss: %.3f' % loss1)
    return accuracy,model


# summarize scores
def model_variability(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


#reformat the data to be in a training set, and a test set.
# the data is being windowed, and the on/off state of the last time is recorded in the one hot vector
trainx, trainy, testx, testy = sep_data(houses)

repeats = 3
scores = list()
for r in range(repeats):
    score,model = run_model(trainx, trainy, testx, testy)
    score *=100.0
    print('>#%d: %.3f' % (r + 1, score))
    scores.append(score)

#how repeatable was the network?
model_variability(scores)




#how well did the model do?

# accuracies of around 70-85% were often achieved (81.5% +-2.3%)  , this is better then an expected accuracy of 65% when predicting off for all times
# less accurate at predictions in house one, predictions were often noisy, and flickered between on off in periodic patterns that were present in the original signal
# this house was expected to be harder as the increase in power was the smallest out of all houses
#house two was predicted the best


# further work
# tune parameters in network and test other window lengths, filter sizes, and layouts more
# could train network taking tv inst power as an input, this could help greatly in pattern recognition in the overall noisy data
# might require additional data especially from house one
# compute confusion matrices to see where false positives, false negatives etc occur
# current windowing method doesnt allow for predictions of the start of the data, include whole data more effectively
#   - could work well for online detection as it uses the last few readings only, not future data
# add in detection of standby mode (as a new class)



