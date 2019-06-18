from sklearn.utils import shuffle as sk_shuffle
import numpy as np

# Image data generator. Applies a random horizontal and vertical shift. It rolls the image, so pixels leaving from end, roll back to the begining of image.
# Keras has a built in generator called ImageDataGenerator, which takes the arguments width_shift_range and height_shift_range to shift the image. When image is shifted to the right, the left pixels are repeated or set to a constant value etc.
# This function rolls back the end of the image to the beginning and vice versa. So we don't need to repeat the pixels.  
# This is not written really to be used for images. This is written for signals that can be rotated. If you rotate an image horizontally, the cat is gonna look weird.
# But if the data is from measurements of linear acceleration, then this will make sense. 
""" 
Why use this? This is similar to vertical/horizontal flipping of an image but it is more general because it 
rolls the data by applying random shift. This is good then the point where we start measurement does not matter. So we can
shift the data in any direction. This helps us in generalizing the model better.
Note that normalization is not supported here. It should be implemented before this step.
"""
def image_generator_ysn(x, y=None, batch_size=64, shuffle=True, enable_shift=True):
    """
    Arguments:
        x: input image of size (n_x, n_h, n_w, n_ch)
        y: second image of same size as x. Use only if the same operation is to be performed on another image y. Most likely should be None.
        batch_size: size of batches of the generator
        shuffle: enable or disable shuffling of data in the beginneing.
        enable_shift: enable or disable vertical/horizontal shift. If disabled, this acts as a simple generator with batch_size.
    """
    if shuffle:
        if y is not None:
            x, y = sk_shuffle(x, y)
        else:
            x = sk_shuffle(x)
    n_x = len(x)
    n_h = x.shape[1]
    n_w = x.shape[2]
    i = 0
    
    # Generator loop
    while True:
        # Get the batch data
        batch_x = x[i: min(i+batch_size, n_x)]
        if y is not None:
            batch_y = y[i: min(i+batch_size, n_x)]

        if enable_shift:
            # horizontal shift by a random number shift
            # We have a slightly higher chance for no shift. See if you can figure out why.
            shift = np.random.randint(-n_w, n_w, 1)
            batch_x = np.roll(batch_x, shift=shift, axis=2)

            # vertical shift
            shift = np.random.randint(-n_w, n_w, 1)
            batch_x = np.roll(batch_x, shift=shift, axis=1)

        # Handle batch increment/end
        i+=batch_size
        if i>=n_x:
            i=0

        # Yield data
        if y is not None:
            yield( batch_x, batch_y )
        else:
            yield( batch_x)

