# peparable-U_Net
# Model 
>>I modified the u_net model named separable-u_net

>> reference the Mobilenet model, I use the separable convolution into unet.

# Environment
>> if you want to implement this repository.the environment need build as follow:

>>>> python==3.6 

>>>> torch==1.1.0

>>>> numpy

>>>> matplotlib

>>>> tensorboardX

# Script interpret

>> the script dataprocess.py is for data read,it's actually a iterable.

>> the script metrics.py is defined miou.

>> the script model_dw.py is the modified unet called separable-unet.

>> the script train.py is for train the model,so when you establish the environment,then can implement this repository in terminal by "python train.py".

>> the script visual_res.py is for visualize the model test result.

# Result
>> I trained 30 epochs.bitch size is 3.

<div align=center><img width="150" height="150" src="https://github.com/2anchao/separable-U_Net/tree/master/images/Train_Loss.svg"/></div>
