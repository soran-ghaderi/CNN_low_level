# CNN_low_level v1.0

### A low-level implementation of Convolutional neural networks (no deep learning framework).

> This version is not Vectorized (therefore slower)

## Methods included:

  | Number  | function           |
  | ------------- | -------------            |
  |       1.     |  `Conv2d forward`             |
  |       2.     | `Conv2d backward `        |
  |       3.     | `maxpool2d forward`  |
  |       4.     | `maxpool2d backward`                |
  |       5.     | `averagepool2d forward`          |
  |       5.     | `averagepool2d backward`          |
  
  Helper funcions:
  
  | Number  | function           |
  | ------------- | -------------            |
  |       1.     | `zero_pad()`             |
  |       2.     | `conv_single_step()`        |
  |       3.     | `distribute_value()`  |
  |       4.     | `create_mask_from_window()`                |
  
  ### Examples
  ```python
  import matplotlib.pyplot as plt
  from nn.layers import *
  
  plt.rcParams["figure.figsize"] = (5.0, 4.0)
  plt.rcParams["image.interpolation"] = "nearest"
  plt.rcParams["image.cmap"] = "Accent"
  
  x = np.random.randn(4, 3, 3, 2)
  x_pad = zero_pad(x, 2)
  print("x.shape =\n", x.shape)
  print("x_pad.shape =\n", x_pad.shape)
  print("x[1,1] =\n", x[1, 1])
  print("x_pad[1,1] =\n", x_pad[1, 1])
  
  fig, axarr = plt.subplots(1, 2)
  axarr[0].set_title('x')
  axarr[0].imshow(x[0, :, :, 0])
  axarr[1].set_title('x_pad')
  axarr[1].imshow(x_pad[0, :, :, 0])
  plt.show()  
  ```
  
  ### Future work
  - [ ] `Add compute cost function`<br/>
  - [X] `Modulate the functions`<br/>
  - [ ] `Add a final model`<br/>
  - [ ] `Add unittest for functions`<br/>
  - [ ] `Vectorize`<br/>
  
  > More documentation will be uploaded later
