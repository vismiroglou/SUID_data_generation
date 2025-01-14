import cv2
import numpy as np

def calc_kernel_size(img:np.ndarray) -> int:
    '''
    Calculate the kernel size for the morphological operations based on the size of the image.
    The paper does not specify this. A large kernel is a way to ensure strong shapes and textures
    don't overpower the image we wish to degrade.
    '''
    num_pixels = img.shape[0] * img.shape[1]
    kernel_size = int(np.sqrt(num_pixels)/20)
    print('Kernel size:', kernel_size, 'x', kernel_size)
    return kernel_size


def calc_backlight(img:np.ndarray, method:str='RCP') -> np.ndarray:
    '''
    Calculate the waterlight/background light of an underwater image with one of three approaches:
    'RCP': Red Channel Prior as outlined in https://www.sciencedirect.com/science/article/pii/S1047320314001874. 
           The backlight is defined as the pixel x0 that maximizes the RCP condition, 
           i.e. min(min(1-I_R(y0), 1-I_G(y0), 1-I_B(y0))) >= min(min(1-I_R(y), 1-I_G(y), 1-I_B(y)))
           where y0 in an area Omega(x0) and y in an area Omega(x) for every x in the image.
    'SUID': The method used in https://ieeexplore.ieee.org/document/9130676,
            The image is repeatedly split into 4 parts, and for every part a score is calculated as the
            difference between the mean and standard deviation of the pixels in the subregion. The subregion
            with the highest score is chosen as the new region until the number of pixels is bellow 1% of the original.
            Finally, the pixels are sorted in descending order based on their intensity and the backlight is chosen as the
            pixel at the 25% mark.
            NOTE: This method does not guarantee that the calculated transmission map values are within the range [0,1].
            It is also not specified by the authors how they enforce that limit (clipping vs. normalization). The results
            show a large variation based on the method used and the example outputs of the paper cannot be replicated.
    'naive': The naive approach is to simply choose the pixel with the highest intensity in the image.
    '''
    
    assert method in ['RCP', 'SUID', 'naive'], 'Invalid method chosen. Choose from RCP, SUID or naive.'

    if method == 'RCP':
        # Define the kernel
        size = (15, 15)
        shape = cv2.MORPH_RECT
        kernel = cv2.getStructuringElement(shape, size)

        # Calculate the red channel prior
        min_b = cv2.erode(img[:,:,0], kernel)/255
        min_g = cv2.erode(img[:,:,1], kernel)/255
        min_r = cv2.erode((255-img[:,:,2]), kernel)/255
        I_red = np.minimum(min_b, min_g, min_r)

        # Get the backlight
        I_red_idx = np.unravel_index(np.argmax(I_red), I_red.shape)
        bl = img[I_red_idx]/255 # BGR format
        print('Backlight:', bl, '\nBacklight Intensity:', np.round(np.mean(bl), 2))
        return bl
        
    elif method == 'SUID':
        # Calculate the backlight
        region = img

        # Calculate threshold
        thr = 0.01 * region[:,:,0].ravel().shape[0]

        # Split the image into 4 parts until the number of pixels is below the threshold
        while region[:,:,0].ravel().shape[0] > thr:
            height, width, _ = region.shape

            center_y, center_x = height // 2, width // 2

            top_left = region[:center_y, :center_x, :]
            top_right = region[:center_y, center_x:, :]
            bottom_left = region[center_y:, :center_x, :]
            bottom_right = region[center_y:, center_x:, :]

            high_score = 0
            for subregion in [top_left, top_right, bottom_left, bottom_right]:
                # Calculate the score for each region as the difference between the mean and the standard deviation
                score = np.mean(subregion) - np.std(subregion)
                if score > high_score:
                    high_score = score
                    region = subregion
        
        # Reshape the region to get the pixels and sort in descending order based on their intensity
        h, w, c = region.shape
        pixels = region.reshape(h * w, c)
        intensities = np.sum(pixels, axis=1)

        sorted_indices = np.argsort(-intensities)
        sorted_pixels = pixels[sorted_indices]

        bl = sorted_pixels[sorted_pixels.shape[0]//4]/255
        print('Backlight:', bl, '\nBacklight Intensity:', np.round(np.mean(bl), 2))
        return bl
    else:
        # This is the naive version where we choose the highest intensity pixel
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        max_loc = np.argmax(img_gray)
        max_loc = np.unravel_index(max_loc, img_gray.shape)
        bl = img[max_loc]/255
        print('Backlight:', bl, '\nBacklight Intensity:', np.round(np.mean(bl), 2))
        return bl


def get_trans_map(img, bl, kernel_size:int=5) -> np.ndarray:
    def calc_coeffs(B_B, B_G, B_R):
        '''
        b_G, b_R and b_B correspond to the scattering coefficients of the green, red and blue channels respectively.
        The papers are not doing a good job at clarifying the units. According to https://ieeexplore.ieee.org/document/7574330 
        the values are '620nm, 540nm and 450nm' which leads to the correct units being nm for the following equations.
        '''
        wl_B = 450
        wl_G = 540
        wl_R = 620
        
        b_B = -0.00113 * wl_B + 1.62517
        b_G = -0.00113 * wl_G + 1.62517
        b_R = -0.00113 * wl_R + 1.62517

        green_red_coeff = b_G * B_R/(B_G * b_R)
        blue_red_coeff = b_B * B_R/(B_B * b_R)
        print('Green-red coeff:', np.round(green_red_coeff, 2))
        print('Blue-red coeff:', np.round(blue_red_coeff, 2))
        return green_red_coeff, blue_red_coeff
    
    # Calculate the transmission map
    size = (kernel_size, kernel_size)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    min_b = cv2.erode(img[:,:,0], kernel)/255
    min_g = cv2.erode(img[:,:,1], kernel)/255
    min_r = cv2.erode((255-img[:,:,2]), kernel)/255

    # This is based on https://www.sciencedirect.com/science/article/pii/S1047320314001874. The SUID paper has the wrong formula.
    b_map = min_b / (bl[0] + 1e-6)
    g_map = min_g / (bl[1] + 1e-6)
    r_map = min_r / (1 - bl[2] + 1e-6)
    
    # At this point a choice needs to be made between clipping and normalizing. The paper does not specify this.
    # b_map = cv2.normalize(b_map, None, 0, 1, cv2.NORM_MINMAX)
    # g_map = cv2.normalize(g_map, None, 0, 1, cv2.NORM_MINMAX)
    # r_map = cv2.normalize(r_map, None, 0, 1, cv2.NORM_MINMAX)

    b_map = np.clip(b_map, 0, 1)
    g_map = np.clip(g_map, 0, 1)
    r_map = np.clip(r_map, 0, 1)
    
    green_red_coeff, blue_red_coeff = calc_coeffs(bl[0], bl[1], bl[2])
    tm_r = 1 - np.minimum(g_map, np.minimum(b_map,r_map))
    tm_r = cv2.blur(tm_r, (kernel_size, kernel_size), 1)
    tm_g = tm_r ** green_red_coeff
    tm_b = tm_r ** blue_red_coeff

    tm = np.array([tm_b, tm_g, tm_r]).transpose(1,2,0)
    return tm


def get_degraded_img(img:np.ndarray, bl:np.ndarray, tm:np.ndarray) -> np.ndarray:
    # BGR
    dimg = img.astype(np.float32) / 255
    dimg = dimg * tm + bl * (1 - tm)
    return dimg 
