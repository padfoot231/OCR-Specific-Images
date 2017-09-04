import glob
import cv2
import numpy as np
import tensorflow as tf
lst = []
for i in range(22):
    l = glob.glob('training set\\class_'+str(i)+'\\*.png')
    lst.append(l)
def y_vector(lst,n):
    y = np.zeros(len(lst))
    for i in range(len(lst)):
        y[i] = n
    return y
lt=[]
for i in range(22):
    l = y_vector(lst[i],i)
    lt.append([lst[i],l])
v = lt[0][1]
l = lt[0][0]
for i in range(21):
    v = np.concatenate((v,lt[i+1][1]),axis = 0)
    l = l + lt[i+1][0]
y_list = v.tolist()
for i in range(len(y_list)):
    y_list[i] = int(y_list[i])
n_values = np.max(y_list) + 1
y = np.eye(n_values)[y_list]
l_image = []
for i in range(len(l)):
    img = cv2.imread(l[i],0)
    img = np.reshape(img,(100,100,1))
    l_image.append(img)
X = np.asarray(l_image)
#X = np.rollaxis(X,0,4)
X_image = np.reshape(X,(X.shape[0],100*100))
print(X_image.shape)
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b 	 
X_image,y = shuffle_in_unison(X_image, y)
X_train = X_image[:5249,:]
y_train = y[:5249,:]
X_val = X_image[5249:,:]
y_val = y[5249:,:]
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
correct_prediction = tf.equal(tf.argmax(y1,1), tf.argmax(y_,1))
prediction = tf.argmax(y_conv,1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#a = np.matrix(h_conv1_1)
saver = tf.train.Saver()
init = tf.global_variables_initializer() 
with tf.Session() as sess:
	sess.run(init)
	for j in range(10):
		l = []
        #print('loss ',sess.run(cost_function,{x:training[:,:30000],y_:training[:,30000:30002],keep_prob:0.5}))
		print('epoch '+ str(j))
        #print('weight ',sess.run(w_fc2[0]))
		for i in range(209):
			x_batch = np.array(X_train[i*25:i*25+25])
            #x_batch = np.multiply(x_batch, 1.0 / 255.0)
			y_batch = np.array(y_train[i*25:i*25+25])
			sess.run(train_step,{x:x_batch,y_:y_batch,keep_prob:0.5})
			if (j+1)%5==0:
				saver.save(sess,'model_Alphabet', global_step=j)
                #print('lol')
			print('loss',sess.run(cost_function,{x:x_batch,y_:y_batch,keep_prob:0.5}))
			l.append(sess.run(accuracy,{x:x_batch,y_:y_batch,keep_prob:0.5}))
		print('mean' , mean(l))
    #print(sess.run(accuracy,{x:x_batch,y_:y_batch,keep_prob:0.5}))
        #print('test accuracy: epoc',j,sess.run(accuracy,{x:X_test,y_:y_test,keep_prob:0.5}))
	ac = []    
	for k in range(58):
		x_batch_test = np.array(X_val[k*10:k*10+10])
			#x_batch_test = np.multiply(x_batch_test,1.0/255.0)
		y_batch_test = np.array(y_val[k*10:k*10+10])
		ac.append(sess.run(accuracy,{x:x_batch_test,y_:y_batch_test,keep_prob:0.5}))
	print('testing accuracy ',mean(ac))
		