import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
from keras import backend as K
import keras
from keras.callbacks import EarlyStopping
# from tensorflow.keras import backend as K
# from tensorflow.python.keras import backend as K

output_size=5 #參數大小
input_size = 8192 #輸入Feature大小
ClassSampleNum = 120 #每個類別的樣本數
f_min = 150
f_max = 70000
Resolution = 140000 / 16384
def getSample(path):
	label = []
	input = []
	with open(path, 'r', encoding='utf-8') as data:
		read = csv.reader(data)
		first_skip=True
		for line in read:
			if first_skip:
				first_skip=False
				continue
			#one_hot=np.zeros(output_size)
			#one_hot[int(line[0])]=1
			
			label.append(int(line[0]))
			raw = []
			for i in line[1::]:
				num=float(i)
				if num>0:
					raw.append(num)
				else:
					raw.append(0)
			raw=np.array(raw)
			raw=raw/np.average(raw)
			input.append(raw)
	return np.array(input),np.array(label)

x,y = getSample("sample.csv")
print(x.shape)
print(y.shape)  


def Reorganize(x):
    x = np.reshape(x, (len(x), 8192))
    x_mean_col = np.mean(x , axis = 0)
    print(x_mean_col)
    x_mean = np.mean(x_mean_col[int((150/Resolution))::])*0.4
    print(x_mean)
    f = -Resolution
    flag = 0
    freq = []
    Idx = []
    #Select region
    print("Select region")
    for i in range(input_size):
        f += Resolution
        if f > f_min and x_mean_col[i] > x_mean:
            if flag == 0:
                flag = 1
                freq.append([f])
                Idx.append([i])
        if f > f_min and x_mean_col[i] < x_mean:
            if flag == 1:
                flag = 0
                freq[len(freq)-1].append(f)
                Idx[len(Idx)-1].append(i)
    
    #merge
    print("merge")
    j=0
    freq = np.array(freq)
    Idx = np.array(Idx)
    for i in range(len(freq)):
        if i == 0:
            continue
        if freq[j+1][0] - freq[j][1] < 300:
            freq[j][1] = freq[j+1][0]
            Idx[j][1] = Idx[j+1][0]
            freq = np.delete(freq, j+1, 0)
            Idx = np.delete(Idx, j+1, 0)
        else:
            j += 1
            
    #Reorganize
    print("Reorganize")
    new_x = []
    for i in range(output_size*ClassSampleNum):
        new_x.append([])
        for idx in Idx:
            new_x[i].extend(x[i][idx[0]:idx[1]])
    
    # plt.plot(new_x[0][0:2369])
    # plt.title("Feature Map")
    # plt.xlabel("Feature point")
    # plt.ylabel("Amplitude")
    # plt.show()
    
    new_x = np.reshape(new_x,(len(new_x), len(new_x[0])))
    print(np.shape(new_x))
    print("Freq. region : ")
    print(freq)
    print(Idx)
    
    # x_label = np.reshape(freq,(len(freq)*2))
    # y_label = np.ones(len(x_label))
    # plt.plot(np.arange(8191)*(Resolution)+Resolution, x_mean_col)
    # plt.bar(x_label,10,100, color='r')
    # plt.plot([0,70000], [x_mean, x_mean])
    # plt.xlabel("Frequency")
    # plt.ylabel("Amplitude")
    # plt.legend(["Signal", "Threshold", "Select_Region"])
    # plt.show()
    return new_x, len(new_x[0])
    
# x, input_size = Reorganize(x)
print(np.shape(x), np.shape(y))
std_x = np.std(x)
mean_x = np.mean(x)
x = (x-mean_x)/std_x
print(std_x, mean_x, x)

train_x=[]
test_x=[]
train_y=[]
test_y=[]
for i in range(len(x)):
    if ((i%ClassSampleNum)>=40 and (i%ClassSampleNum)<60):
        test_x.append(x[i])
        test_y.append(y[i])
    elif (i%ClassSampleNum)<ClassSampleNum:
        train_x.append(x[i])
        train_y.append(y[i])
train_x=np.array(train_x)
train_y=np.array(train_y)
test_x=np.array(test_x)
test_y=np.array(test_y)

# in order to plot in a 2D figure
encoding_dim = 2

loss_set = []
layer_set = []
zoom_set = []
lr_set = []
node_set = []
activation_set = []
loss_func_set = []
# GS parameter:layer zoom lr activation loss_func
for Node in [4096,2048,1024,512,256,128,65,32,16,8,4,2,1]:
    for LR in [0.001,0.01]:
        for Activation in ['relu','sigmoid','tanh']:
            for Loss_func in ['mse','mae']:
                print("Node:", Node, "LR:", LR ,"Activation:", Activation, "Loss_func:", Loss_func)
                K.clear_session()
                input_img = Input(shape=(input_size,))
                encoded = Dense(Node, activation=Activation)(input_img)
                # BTN = BatchNormalization(axis=-1, epsilon=0.001, center=True)(encoded)
                decoded = Dense(input_size)(encoded)
                # neuron = 400
                # encoder layers
                # for i in range(Layer_Num):
                    # neuron = int(neuron/Zoom)
                    # if i == 0:
                        # encoded = Dense(neuron, activation=Activation)(input_img)
                    # else:
                        # encoded = Dense(neuron, activation=Activation)(encoded)
                # decoder layers
                # for i in range(Layer_Num):
                    # neuron = int(neuron*Zoom)
                    # if (i == (Layer_Num-1)) and (i == 0):
                        # decoded = Dense(400)(encoded)
                    # elif i == (Layer_Num-1):
                        # decoded = Dense(400)(decoded)
                    # elif i == 0:
                        # decoded = Dense(neuron, activation=Activation)(encoded)
                    # else:
                        # decoded = Dense(neuron, activation=Activation)(decoded)

                # construct the autoencoder model
                autoencoder = Model(input=input_img, output=decoded)
                print(autoencoder.summary())

                # construct the encoder model for plotting
                # encoder = Model(input=input_img, output=encoder_output)

                # compile autoencoder
                adam = Adam(lr = LR)
                autoencoder.compile(optimizer=adam, loss=Loss_func)

                # training
                ES_Acc = EarlyStopping(monitor='val_loss',min_delta=0, mode='min', verbose=1, patience=200)
                history = autoencoder.fit(train_x, train_x, epochs=10000, batch_size=256, shuffle=True,callbacks=([ES_Acc]), validation_data=(test_x, test_x))
                loss_set.append(min(history.history['loss']))
                # layer_set.append(Layer_Num)
                # zoom_set.append(Zoom)
                node_set.append(Node)
                lr_set.append(LR)
                activation_set.append(Activation)
                loss_func_set.append(Loss_func)
                with open('GS_AE_0708_5p_STD.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Set', 'loss_func','Node','LR','Activation','loss_Func'])
                    for i in range(len(loss_set)):
                        writer.writerow([i+1,  loss_set[i], node_set[i], lr_set[i], activation_set[i], loss_func_set[i]])
                

# plotting
# encoded_imgs = encoder.predict(x)
# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y)
# plt.colorbar()
# plt.show()