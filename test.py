import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv
from keras import backend as K
import keras
from keras.models import load_model
import tensorflow as tf
# def getSample(path):
	# label = []
	# input = []
	# with open(path, 'r', encoding='utf-8') as data:
		# read = csv.reader(data)
		# first_skip=True
		# for line in read:
			# if first_skip:
				# first_skip=False
				# continue
			# #one_hot=np.zeros(output_size)
			# #one_hot[int(line[0])]=1
			
			# label.append(int(line[0]))
			# raw = []
			# for i in line[1:]:
				# num=float(i)
				# if num>0:
					# raw.append(num)
				# else:
					# raw.append(0)
			# raw=np.array(raw)
			# raw=raw/np.average(raw)
			# input.append(raw)
	# return np.array(input),np.array(label)

# x,y = getSample("sample_14L_Amptitute_0529.csv")
# # x,y = getSample("sample_mid.csv")
# print(np.shape(x), np.shape(y))


output_size=5 #類別數
input_size = 8192 #輸入Feature大小
ClassSampleNum = 240 #每個類別的樣本數
TestSetNum = 40
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
std_x = np.std(x)
mean_x = np.mean(x)
x = (x-mean_x)/std_x
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

def Greedy(evaluate_result, Model_num, sample_num, G_index = None, Threshold = None):
    TP=0
    TN=0
    FP=0
    FN=0

    if Threshold == None:
        sort_lose = np.sort(evaluate_result)
        Threshold = sort_lose[Model_num][G_index]
    
    for i,lose_ in enumerate(evaluate_result[Model_num]):
        if lose_ > Threshold: ##樣本lose大於當前閥值，即判定為不合格
            if (i%len(evaluate_result[Model_num]) >= Model_num*sample_num) and (i%len(evaluate_result[Model_num]) < (Model_num+1)*sample_num): ##實際為合格
                FP += 1
            else:##實際為不合格
                TP += 1
        else: ##樣本lose小於當前閥值，即判定為合格
            if (i%len(evaluate_result[Model_num]) >= Model_num*sample_num) and (i%len(evaluate_result[Model_num]) < (Model_num+1)*sample_num): ##實際為合格
                TN += 1
            else: #實際為不合格
                FN += 1
    TPR = TP/(FN+TP)
    FPR = FP/(FP+TN)
    PRE = TP/(TP+FP)
    ACC = (TP+TN)/(TP+TN+FP+FN)
    
    return TPR, FPR, PRE, ACC, TP, TN, FP, FN

pred_Data_test = []
pred_Data_train = []
for i in range(output_size*ClassSampleNum):
    if (i%ClassSampleNum >= ClassSampleNum-TestSetNum):
        pred_Data_test.append([x[i]])
    else:
        pred_Data_train.append([x[i]])
pred_Data_test = np.array(pred_Data_test)
pred_Data_train = np.array(pred_Data_train)
print(np.shape(pred_Data_test))

##讀取所有模型
class_model = []
for i in range(output_size):
    class_model.append(load_model('./AE_Model/model_'+repr(i)+'/model_'+repr(i)+'.h5'))
    # class_model.append(tf.keras.models.load_model("model_"+repr(i)+".h5"))
    
##跑lose
evaluate_result_test = []
evaluate_result_train = []
for i in range(output_size):
    cc = []
    for j in range(output_size*TestSetNum):
        cc.append(class_model[i].evaluate(pred_Data_test[j],pred_Data_test[j])) ## test set
    evaluate_result_test.append(cc)
    
    cc = []
    for j in range((ClassSampleNum-TestSetNum)*output_size):
        cc.append(class_model[i].evaluate(pred_Data_train[j],pred_Data_train[j]))  ## train set
    evaluate_result_train.append(cc)
print("loss :", evaluate_result_train[0][0])