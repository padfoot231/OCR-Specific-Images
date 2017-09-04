import glob
import cv2
import numpy as np
import tensorflow as tf
from operator import itemgetter
import difflib
import os
import shutil
from shutil import copyfile
MIN_CONTOUR_AREA = 10
lst = glob.glob('*.png')
l_img = []
for i in range(len(lst)):
    l_ROI = []
    img = cv2.imread(lst[i],0)
# inverting pixels of image to make contours
    img_thresh = 255 - img 
#plotting contours
    img_contours,n_contours,n_hierarchy = cv2.findContours(img_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for j in range(len(n_contours)):
        if cv2.contourArea(n_contours[j])>MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(n_contours[j])
            cv2.rectangle(img,(intX, intY),(intX+intW,intY+intH),(255, 255, 255),1)
            imgROI = img[intY:intY+intH, intX:intX+intW]
            l_ROI.append([intX,imgROI])
# l _img contains all the segregated images of each image in a document along with the x co-ordinate of top right corner
# l_img[i] gives list of all segregated characters of ith image of a document  along with it x co-ordinate
# l_img[i][1][0] gives matrix of first character of ith image in a document
    l_img.append(l_ROI)
l_sorted = []
for i in range(len(l_img)):
    temp = sorted(l_img[i],key=itemgetter(0)) 
    l_sorted.append(temp)
#l_sorted list contains sorted images according to there top right x co-ordinate
lt = []
#padding of images in the list with zeros to make the size 100X100
for i in range(len(l_sorted)):
    for j in range(len(l_img[i])):
        l_sorted[i][j][1] = 255 - l_sorted[i][j][1]
        l_sorted[i][j][1] = cv2.resize(l_sorted[i][j][1],(30,50))
        npad = ((25,25),(35,35))
        l_sorted[i][j][1] = np.pad(l_sorted[i][j][1], pad_width=npad, mode='constant', constant_values=0)
l_image_doc = []
for i in range(len(l_sorted)):
    l_image = []
    for j in range(len(l_sorted[i])):
        l_sorted[i][j][1] = np.reshape(l_sorted[i][j][1],(100,100,1))
        l_image.append(l_sorted[i][j][1])
    l_image_doc.append(l_image)
l_array = []
for i in range(len(l_image_doc)):
    X = np.asarray(l_image_doc[i])
    X_image = np.reshape(X,(X.shape[0],100*100))
    l_array.append(X_image)
# l_array contains list of all the images in a document converted into array of size (number of characterX10000)
def Weights(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)
def bias(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
x = tf.placeholder(tf.float32,[None,10000])
y_ = tf.placeholder(tf.float32,[None,22])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x,[-1,100,100,1])
def conv(x,W,s_c):
    return tf.nn.conv2d(x,W,strides=[1,s_c,s_c,1],padding ='VALID')
def max_pool(x,s_p,k_s):
    return tf.nn.max_pool(x,ksize=[1,k_s,k_s,1],strides=[1,s_p,s_p,1],padding='SAME')
def mean(l):
    return sum(l)/len(l)
#conv1_1
w_conv1_1 = Weights([11,11,1,96])
b_conv1_1 = bias([96])
h_conv1_1 = tf.nn.relu(conv(x_image,w_conv1_1,3)+b_conv1_1)
# maxpool
h_pool1 = max_pool(h_conv1_1,2,3)
#conv1_2
w_conv1_2 = Weights([5,5,96,256])
b_conv1_2 = bias([256])
h_conv1_2 = tf.nn.relu(conv(h_pool1,w_conv1_2,1)+b_conv1_2)    
#maxpool
h_pool2 = max_pool(h_conv1_2,2,3)
    #conv2_1
w_conv2_1 = Weights([3,3,256,256])
b_conv2_1 = bias([256])
h_conv2_1 = tf.nn.relu(conv(h_pool2,w_conv2_1,1)+b_conv2_1)
#maxpool
h_pool3 = max_pool(h_conv2_1,2,3)   
    #fully connected layer
w_fc1 = Weights([2*2*256,1024])
b_fc1 = bias([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 2*2*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, w_fc1) + b_fc1)
    #dropout
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    #softmax layer
w_fc2 = Weights([1024,22])
b_fc2 = bias([22])
y_conv = tf.matmul(h_fc1_drop,w_fc2)+b_fc2
y1 = tf.nn.softmax(y_conv)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cost_function)
#correct_prediction = tf.equal(tf.argmax(y1,1), tf.argmax(y_,1))
prediction = tf.argmax(y_conv,1)
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#a = np.matrix(h_conv1_1)
saver = tf.train.Saver()
init = tf.global_variables_initializer() 
with tf.Session() as sess:
    sess.run(init)
    ac = []
    saver = tf.train.import_meta_graph('model_Alphabet-9.meta')
    sess.run(tf.global_variables_initializer())
    #tf.train.Saver.restore(sess=sess,save_path=')
    saver.restore(sess,tf.train.latest_checkpoint('.\\'))    
    l_str = []
    for i in range(len((l_array))):
        str1 = []
        for j in range(l_array[i].shape[0]):
            #x_batch = np.reshape(l_array[i][j,:],(1,10000))
            l = (sess.run(prediction,{x:l_array[i],keep_prob:0.5}))
            if l[j] == 0:
                str1.append('A')
            if l[j] == 1:
                str1.append('a')
            if l[j] == 2:
                str1.append('B')
            if l[j] == 3:
                str1.append('c')
            if l[j] == 4:
                str1.append('d')
            if l[j] == 5:
                str1.append('e')
            if l[j] == 6:
                str1.append('E')
            if l[j] == 7:
                str1.append('f')
            if l[j] == 8:
                str1.append('H')
            if l[j] == 9:
                str1.append('I')
            if l[j] == 10:
                str1.append('m')
            if l[j] == 11:
                str1.append('n')
            if l[j] == 12:
                str1.append('N')
            if l[j] == 13:
                str1.append('o')
            if l[j] == 14:
                str1.append('P')
            if l[j] == 15:
                str1.append('R')
            if l[j] == 16:
                str1.append('r')
            if l[j] == 17:
                str1.append('S')
            if l[j] == 18:
                str1.append('t')
            if l[j] == 19:
                str1.append('u')
            if l[j] == 20:
                str1.append('v')
            if l[j] == 12:
                str1.append('y')    
        s = ("".join(str1))
        l_str.append(s)
    #str2 = 'test'
    str1 = ['beneficiarysnameandaddress','panno','commodity',
'remmitersnameandaddress','valuedate','valuedateapplicableonlyifvaluedateintheswift','iecode', 'hscode']
    #seq=difflib.SequenceMatcher(None, str2,str1[i])
    for i in range(len(l_str)):
        d_max=0
        for j in range(8):
            a = l_str[i]
            str2 = a.lower()
            #print(str2)
            seq=difflib.SequenceMatcher(None, str2,str1[j])
            d=seq.ratio()*100
            #print(d)
            if d>d_max:
                d_max = d
                index = j
        #print(d_max)
        if d_max>70:
            if index == 0:
                print('beneficiary\'s name and address' )
                if not os.path.exists('beneficiary'):
                    os.mkdir('beneficiary')
                shutil.copy(lst[i],'beneficiary')
            if index == 1:
                print('PAN No')

                if not os.path.exists('pan'):
                    os.mkdir('pan')
                shutil.copy(lst[i],'pan')
            if index == 2:
                print('Commodity')
                if not os.path.exists('commodity'):
                    os.mkdir('commodity')
                shutil.copy(lst[i],'commodity')
            if index==3:
                print('Remmiter\'s name and address')
                if not os.path.exists('remmiter'):
                    os.mkdir('remmiter')
                shutil.copy(lst[i],'remmiter')
            #if index==4:
             #   print('Purpose of remmitence')
            if index == 4 or index == 5:
                print('value date')
                if not os.path.exists('value date'):
                    os.mkdir('value date')
                shutil.copy(lst[i],'value date')
            if index==6:
                print('I.E.Code')
                if not os.path.exists('I.E.Code'):
                    os.mkdir('I.E.Code')
                shutil.copy(lst[i],'I.E.Code')
            if index==7:
                print('H.S.Code')
                if not os.path.exists('H S Code'):
                    os.mkdir('H S Code')
                shutil.copy(lst[i],'H S Code')
        else:    
            if not os.path.exists('garbage'):
                os.mkdir('garbage')
            shutil.copy(lst[i],'garbage')       