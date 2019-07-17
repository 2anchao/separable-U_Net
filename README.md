# Peparable-U_Net
## Model 
>>I modified the u_net model named separable-u_net

>> reference the Mobilenet model, I use the separable convolution into unet.

## Environment
>> if you want to implement this repository.the environment need build as follow:

>>>> python==3.6 

>>>> torch==1.1.0

>>>> numpy

>>>> matplotlib

>>>> tensorboardX

## Script interpret

>> the script dataprocess.py is for data read,it's actually a iterable.

>> the script metrics.py is defined miou.

>> the script model_dw.py is the modified unet called separable-unet.

>> the script train.py is for train the model,so when you establish the environment,then can implement this repository in terminal by "python train.py".

>> the script visual_res.py is for visualize the model test result.

## Result
>> I trained 30 epochs.bitch size is 3.
>> Train loss and Train miou：

![Train loss and Train miou](images/1.png)

Valid loss and Valid miou：

![Valid loss and Valid miou](images/2.png)

## Visual
>> the segmentation result:

![segmentation result1](save_visual/1.jpg)

![segmentation result2](save_visual/2.jpg)

![segmentation result3](save_visual/3.jpg)

## Analysis
>> There are still many improvements can use in this reporsitory, I just attempt to use separable convolution, dilate convolution and residual structure to unet. The result is better than original unet in my datasets. And the final use of the model is linear output, adding the sigmiod activate function may be better. The epoch of model training is less, and the effect of continuing training may be better.

## Attention
>> The project was completed by me independently for academic exchange. For commercial use, please contact me by email an_chao1994@163.com.


