The code works as OCR. It segregate all the characters in a gray scale text image and predict the text written on it based on the predicted characters. This code only works for specific text images, hence there are only 22 classes of character instead of 62 classes.

Instructions to run training.py and predict.py

Training:	
Anaconda Prompt:
>> activate tensorflow
>> python train.py

Testing: 
To provide input in code predict.py go to line number 11 and give the path of the document containing all .png files as argument to glob.glob() function

Anaconda Prompt:
>>activate tensorflow
>> python predict.py

train.zip contains training set with 22 classes of alphabets 

predict.zip contains text images to be predicted
