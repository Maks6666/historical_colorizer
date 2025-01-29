# Image colorization with advanced PyTorch autoencoders.

![title.png](content%20/title.png)

Being obsessed as with history, as with programming I was always interesting in finding a way to make old gray photos 
colorized. Such neural network type as autoencoder provides such possibility, so I decided to research this topic. 
Previously, I already realized a model for gray image colorization with Tensorflow model, but this one must work much 
better because a lot of improvements, which were provided using PyTorch framework.

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

More information about training process is available in 'correct_colorizer_training.ipynb' file.

### Metrics 


### Deployment


