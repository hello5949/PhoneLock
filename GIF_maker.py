import os
import imageio
images = []

path = './GS_Result'
filenames = os.listdir(path)
filenames.sort()
print(filenames)
for filename in filenames:
    if filename.endswith('.png'):
        images.append(imageio.imread(path+'/'+filename))
        print(filename)
imageio.mimsave('./result_MSE.gif', images)