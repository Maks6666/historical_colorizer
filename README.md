# Image colorization with advanced PyTorch autoencoders.

![title.png](content%20/title.png)

Being obsessed as with history, as with programming I was always interesting in finding a way to make old gray photos 
colorized. Such neural network type as autoencoder provides such possibility, so I decided to research this topic. 
Previously, I already realized a model for gray image colorization with Tensorflow model, but this one must work much 
better because a lot of improvements, which were provided using PyTorch framework. Moreover, an image dataset for there models, 
which contains more that 12.000 images, I collected by myself. 

### LAB-Apporach 

Unlike the previous model, this one doesn't work with RGB-formatted images. The main idea in case of preparation of data 
for this type's models is, to put colorized images into LAB format from RGB and split it into L and AB image channels as 
features and labels. In a LAB format L-channel means a brightness of an image and contains values from 0 to 100, 
where 0 represents absolute black (complete absence of light), and 100 represents maximum brightness (pure white). 
So L-channel of any image (colorized or not) will always show the gray form of this image. This will be taken as 
a feature for the model, based on which, model should return colorized channels of image - or AB channels. 
A-channel in LAB format represents color information on the green-red axis, 
where negative values indicate green tones, and positive values indicate red tones. B-channel in LAB format represents color information on the blue-yellow axis, 
where negative values indicate blue tones, and positive values indicate yellow tones. Both of them contains information in values from
~ -128 to ~127. So, assuming that, AB channels contains information about real colors of grayscale image (L-channel) and
that is why they will be used in my model as labels, which will be predicted ny autoencoder. 

How to turn image to LAB format and "cut" channels from it:

```
from PIL import Image
from scimage.color import rgb2lab
from torchvisiion import transforms


def channels_extractor():

    tramsformer = transforms.Compose([
        transforms.ToTensor()
    ])
    
    
    link = "link/to/my/image"
    image = Image.open(link).convert("RGB")
    
    if tramsformer:
        image = tramsformer(image)
    
    # correct shape of np_image mush be: (H, W, channels)
    np_image = image.permute(1, 2, 0).cpu().numpy()
    lab_image = rgb2lab(np_image)
    
    l_channel = lab_image[:, :, 0] / 100.0
    ab_channels = lab_image[:, :, 1:] / 128.0
    
    l_channel = torch.tensor(l_channel, dtype=torch.float32).unsqueeze(0)
    ab_channels = torch.tensor(ab_channels, dtype=torch.float32).permute(2, 0, 1)
    
    return l_channel, ab_channels
    
``` 
IMPORTANT: during channel extraction with  'lab_image[:, :, 0]' and 'lab_image[:, :, 1:]' I also applied a normalization
to channel values to put l_channel tensor in 0-1 range and ab_channels tensor in -1-1 range. That's why model prediction
will be in range from -1 to 1. This requires 'tanh' activation function usage on the last output layer.


And if you need, to put separated l_channel and ab_channels back into s single RGB-image, then you must do it this way:


```
import numpy as np
from skimage.color import lab2rgb


def rgb_image(l_channel, ab_channels):

    l_channel = l_channel.permute(1, 2, 0).squeeze(2).cpu().numpy()
    ab_channels = ab_channels.permute(1, 2, 0).cpu().numpy()
    
    a_channel = ab_channels[:, :, 0]
    b_channel = ab_channels[:, :, 1]
    
    # then it needs to apply clipping process on receied arrays to make sure that all arrays will be in correct ranges.
    # it is also necessary no normilize all arrays in necessary ranges (0-100 for l_channel and -128-127 for a/b-channel).
    
    l_channel = np.clip(l_channel * 100, 0, 100).astype(np.uint8)
    a_channel = np.clip(l_channel * 128, -128, 127).astype(np.int8)
    b_channel = np.clip(b_channel * 128, -128, 127).astype(np.int8)
    
    # make sure all of arrays have size: H, W, 1 before uniting them back:
    
    lab_image = np.stack([l_channel, a_channel, b_channel], axis=-1).astype(np.float64)
    
    rgb_image = lab2rgb(lab_image)
    
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    # rgb_image - is basic version of your image.
    
    
``` 

### Model structure 
![predictions.png](content%20/predictions.png)

As it was mentioned, model will have one neuron on the input layer, which represents l-channel and two neurons on the 
last, output layer, which will be combined with l-channel from input in predict method and converted to ready RGB-image.

To provide an effective performance and high accuracy model requires several structural features, which must increase it's 
quality:

![structure.png](content%20/structure.png)

- Basic model implementation has 6 convolution layers (nn.Conv2d), including bottleneck one, which put tensor into 512, 16, 16 size. 
After last convolution layer model has 4 convolution-transpose (nn.ConvTranspose2d) layers, which increase tensor size to basic form. Each 
convolution-transpose layer is united with convolution layer with same tensor shape using tensor concationation method - torch.cat(...). This move copies part of U-net
structure and helps to increase models possibility to reconstruct convoluted information. Layer, which comes before 
the last one concatenates tensor with an input tensor and passes tensor of size: 3, H, W on the last layer, which is also 
a convolution one (nn.Conv2d), which return 2, H, W tensor as it was mentioned and does it using tanh() activation function, 
which puts all values in -1-1 range as I normalize a/b channels: 

```
          E^x - E^-x
tanh(x) = ----------
          E^x + E^-x
```

- After all of convolution and convolution-transpose layers, model has a batch normalization layers (nn.BatchNorm2d) for better
regulation. 
- Bottleneck layers contains dilation kernel - parameter, which increases an area, where convolution kernel can find 
features without changing kernel size. It helps model to extract complicated features from convoluted image.
- Model also uses memory-mechanisms to memorize such features as color, which are important in colorization task. It uses
a squeeze-excitation layer after each convolutional and convolution-transpose layer, which also improves model accuracy. Basic idea 
if squeeze-excitation layer is to evaluate an importance of each filer, turn it into number-array, multiply feature map 
on each number of the array thus, that more important ones will be multiplied on bigger values and less important ones 
will be multiplied on fewer values. This could be realised this way:

```

class SEBlock(nn.Module):
    def __init__(self, channels, division_coef=16):
        super().__init__()
        
        # to extract averaged importand information from each filter (channel) I use AdaptiveAvgPool2d:
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        
        # AdaptiveAvgPool2d returns a tensor with shape of: ('batch, channels, 1, 1' - one pixel contains averaged info from each filter), 
        # so I use Flatten to put it into vector of size (batch, channels*1*1) where each element will represent an importance of each filter:
        self.flatten = nn.Flatten()
        
        # I change the shape of above mentioned vector with Linear layers to make SEBlock not to concantrate on any specific static shape:
        self.linear1 = nn.Linear(channels, channels//division_coef)
        self.linear2 = nn.Linear(channels//division_coef, channels)
        
        # I use relu to delete possible negative values:
        self.relu = nn.ReLU()
        
        # And sigmoida will put final 'importance-container' vector into (0, 1) range:
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.aap(x)
        out = self.flatten(out)
        
        out = self.linear1(out)
        out = self.relu(out)
        
        out = self.linear2(out)
        out = self.sigmoid(out)
        
        out = out[:, :, None, None]
        
        # finally, 'out' - is a tensor of size (batch, channels, 1, 1) each element of which represents an importance
        # of all corresponding channels from original tensor - 'x'. That's why I finally multiply original one with 
        # an 'importance container' - to make important channels bigger (beacuse they will be multiplied on bigger values from 'importance container')
        # and to make less important channels fewer (beacuse they will be multiplied on lower values from 'importance container').
        
        res = out * x
        
        return res 
```

### Knowledge distillation: 
Having so advanced structure, final weight of model is pretty heavy:
![teacher_info.png](content%20/teacher_info.png)

But this issue could be solved with knowledge distillation training mechanism, which trains a smaller (student) model, based 
on logits, received from a bigger (teacher) model. But in case of autoencoder neural network it works in a specific way.
Unlike as in convolutional networks, it doesn't need to calculate KL-divergention (at least in this case) to calculate 
a soft losses, which in case of autoencoder training are named as distillation loss - a loss between latent outputs
of student and teacher models. Latent output - is and output of bottleneck layer, when pass-tensor is in the most 
convoluted form. IMPORTANT: latent layers of both models should be the same. It is possible to set latent output this way:

```
class Autoencoder:
    ...
    
    def forward(self, x, return_latent=False):
    out = self.layer(x)
    ...
    if return_latent == True:
        return out
    ...
    out = self.layer_n(out)

```

To calculate a loss between teacher and student latent output it is enough to use a MSE loss function:

```
1/n*∑(y_true - y_pred)^2

# example:

y_true = [6, 5, 4]
y_pred = [3, 2, 1]

1) [6, 5, 4] - [3, 2, 1] = [3, 3, 3]
2) [3, 3, 3] ^ 2 = [9, 9, 9]
3) (9 + 9 + 9) / 3 = 9

MSE = 9

```

Calculating MSE between student and teacher latent output, we need also to define a reconstruction loss, which is 
actually just a normal loss between model predictions and labels - here we also could use a MSE. And then it is possible
to find a new loss:

```
loss = reconstruction_loss * alpha + distillation_loss * beta
```
Where alpha and beta - are weights, which regulate weights of reconstruction and distillation losses. By default they both are 
0.5, but to give a priority for some parameter, it is possible to increase a value of correspondent weight (alpha for reconstruction_loss and
and beta for distillation_loss). The whole distillation training process could be defined this way:

```

loss_fn = nn.MSELoss()
dist_fn = nn.MSELoss()

def training_loss(train_dl, student_model, teacher_model, optimizer, loss_fn, dist_fn, epochs, alpha=0.5, beta=0.5, ...):
    ...
    
    
    for epoch in range(epochs):
        teacher_model.eval()
        
        train_loss = 0.0
        
        for x, y in train_dl:
            
            with torch.no_grad():
                teacher_latent = teacher_model(x, return_latent=True)
            
            student_latent = student_model(x, return_latent=True)
            student_reconstruct = student_model(x, return_latent=False)
            
            distillation_loss = dist_fn(student_latent, teacher_latent)
            reconstruct_loss = loss_fn(student_reconstruct, y)
            
            loss = distillation_loss * alpha + reconstruct_loss * beta
            train_loss += loss
            
            ...
           


```
So as a result, knowledge distillation allowed to decrease model size into two times - student model doesn't have a 
U-net element or memorizing mechanism - it is just simple autoencoder, but it works not worse, than a original model.
Student model parameters:

![student_info.png](content%20/student_info.png)

And as a result, losses of both models, look pretty good, which is actually proved by not bad performance of both models:
![losses.png](content%20/losses.png)

More information about training process is available in 'correct_colorizer_training.ipynb' file.

### Metrics 

During both models training, I used two metrics, except MSE, to control an accuracy of image colorization - PSNR and SSIM:
- PSNR (peak signal-to-noise ratio) - metric, which calculates differance between two images - original one 
(grayscale image of l_channel in this case) and one, returned by neural network (colorized one):

```
PSNR = 10 * log(10) * (MAX^2 / MSE)
```
where:
MAX^2 - maximum value of pixels (1 in case, when they are normalized in -1-1 range)
MSE - mean squared error between original images and predicted one. 

In my case I applied PSNR to find a similarity between predicted AB channels and original ones and that works pretty well in both models.
It was necessary only to specify, that my data_range of PSNR is equal to 2. PSNR values for my models:

![psnr.png](content%20/psnr.png)


- SSIM (structural similarity index measure) - metrics, which is able to calculate differance between images 
taking into account an interpretation ob image by human-eye. Unlike PSNR, which just calculate differance between each pixel, 
SSIM is able to consider structural differance between images. In other worlds it is able to evaluate whole image - not just each pixel separately. 
It works with a next formula:

![ssim.png](content%20/ssim.png)


Where:

1) μx, μy - mean values of brightness of some windows (local areas) of images x and y. They could be found this way:
```
μ_n = 1/N ∑(I_n)

# I_n - is a value of each pixel in window
# N - amount of pixels in window

# for example, having n image of size 2x2:

# [10, 20,
# 30, 40]

# μ_n will be:

# (10 + 20 + 30 + 40) / 4 = 100 / 4 = 25 

# But if we've got an image of size 16x16, and size of window will be manually setted as 4x4, each 4x4 area will be a 
# a window (local area) - in this case we need to calculate μ of each window and find their mean value. 
```

2) σx, σy - mean dispersion (contrast) of all windows (local areas). They could be found this way: 
```
σ_n = 1/N ∑(I_n - μ_n)^2

# I_n - value of each pixel in window 
# μ_n - mean value of brightness in a correspondent window
# N - amount of pixels in window

# In the same way, if you've got a window n of size:

# [10, 20,
# 30, 40]

# σ_n = ((10 - 25)^2 + (20 - 25)^2 + (30 - 25)^2 + (40 - 25)^2) / 4 = (225 + 25 + 25 + 225) / 4 = 125

# And in the same way, if you've got several windows with specific sizes, then you need to calculate 
# σ of each of them and and find a mean value. 
```
3) σx_y = covariation between x and y images, which calculates structural similarity between images x and y.
It calculates this way:

```
σx_y = 1/N ∑(I_x - μ_x) * (I_y - μ_y)

# I_x - value of each pixel in x image specified window
# μ_x - mean value of brigntness in x specified windown

# I_y - value of each pixel in y image specified window
# μ_y - mean value of brigntness in y specified windown

#N - amount of pixel in each of both windows

# In a similar way, if you've got two images - x and y with a same size:

# [10, 20,
# 30, 40]

# σx_y = ((10 - 25) * (10 - 25)) + ((20 - 25) * (20 - 25)) + ((30 - 25) * (30 - 25)) + ((40 - 25) * (40 - 25)) / 4 =
# = (225 + 25 + 25 + 225) / 4 = 125

# And in the same way, if you've got several windows with specific sizes, then you need to calculate 
# σx_y of each of them and and find a mean value.
```

4) C1, C2 - small values, which prevent zero division. Usually:
```
C1 = (0.01 * L)^2
C2 = (0.03 * L)^2

# Where L - is maximum pixel value.
```

IMPORTANT: SSIM accepts only 1D values, so in my case, during applying metrics on predicted and true AB channels, 
which are actually 2D data, i find mean values between predicted and true A and B channels and only then fit it to the
SSIM:

```
preds_np = preds.cpu().detach().numpy()
y_np = y.cpu().detach().numpy()

# here it means that preds_np shape is: (B, C, H, W), so C - channel is under index 1

if preds_np.shape[1] > 1:  
   preds_np = preds_np.mean(axis=1) 
   y_np = y_np.mean(axis=1)
            
   t_ssim = ssim(preds_np.squeeze(), y_np.squeeze(), win_size=7, data_range=2.0)
```
SSIM values for my models: 

![ssim2.png](content%20/ssim2.png)

### Deployment
You may test both models by yourself with the main.py file. All you need to do is only to put gray images you'd like to colorize, 
choose model with a latter in string:

```
colorizer = ColorizerApp(image, dir_to_save, "l")'
```

where 'l' means 'large', so a teacher model, and 's' means 'small', so a student model. Directory to save colorized 
image is specified by default as 'colorized_image' - of course you may change it path if it is necessary. To start colorization process, 
as I said, you need to run a file, then input an index of an image you'd like to colorized, which will be displayed in a terminal. 
After input, program will open image with PIL, resize it with torchvision.transforms, put into specific size, extract l_channel
in above mentioned way and pass it through model. Method predict will automatically combine ab and l channels into single image and 
turn it to "RGB" format, so you will be able to find it's colorized version in 'colorized_image' folder. 

### Addition

This project also includes one more model, which could be called with "A" so an additional one instead of 'l' or 's'.
The specific of this model is, that it has a sigmoid() function on the output layer instead of tanh(), so it return's
a/b channels in range of 0 and 1. And despite being wrong from theoretical point of view, this model also is available and 
I must admit, that this one works really not bad. For more additional information about it, you could check file "colorized_training.ipynb".
Other methodic in this one is similar as in above mentioned models.

### Video colorization
This program is also able to adapt colorization models for video files colorization. To do that, you just need to specify model type by 
same model marker as in main.py: l - for teacher model, s - for student mode, a - for additional model. Then, run a file and the same way 
as for main.py, choose video, you'd like to colorize from 'grayscaled_videos' folder by input it's index. This video will be splitted into separate frames, which will be turned into dataset and dataloader
to be passed through the model. Model will colorized each of separated frames. Results in tensor form will be saved to an array and then will be written 
into new video file using cv2.VideoWriter() method. Colorized video will be saved in .mp4 format into 'colorized_videos' folder.

Thanks for attention!

### Autors:
- Kucher Maks (maxim.kucher2005@gmail.com / Telegramm (for contacts): @aeternummm)

