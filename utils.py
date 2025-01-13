import cv2
import numpy as np

'''Notes:
THis method does not use the dark channel prior in its original format because
it correctly assumes that in underwater environments not all channels are equal.
It bases everything on the assumption that the red channel will more often 
than not be the most degraded one. This does not hold for the simulated data that is in 
super shallow waters.
'''

B_R = 0.11
B_G = 0.31
B_B = 0.56
T_R = 0.20
T_G = 0.55
T_B = 0.72

def calc_backlight(img, thr=10):
    # Calculate the backlight
    region = img

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

    # Add an intensity threshold
    # This is not done in the paper. In fact, the intensity of the backlight has a very large effect on the final result. 
    # High intensity values of the backlight will inevitably result in a very dark transmission map, which in turn means that the
    # shapes of the underwater image will overpower the image we wish to degrade
    # if np.mean(bl) > 0.4:
    #     bl = bl - (np.mean(bl) - 0.4)
    # This is the naive version where we choose the highest intensity pixel
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # max_loc = np.argmax(img_gray)
    # max_loc = np.unravel_index(max_loc, img_gray.shape)
    # bl = img[max_loc]/255
    
    # This is using an example value from the paper
    # bl = [B_B, B_G, B_R]

    print('Backlight:', bl, '\nBacklight Intensity:', np.mean(bl))
    return bl


def get_trans_map(img, bl):
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
        print('Green-red coeff:', green_red_coeff)
        print('Blue-red coeff:', blue_red_coeff)
        return green_red_coeff, blue_red_coeff
    
    # Calculate the transmission map
    size = (9, 9)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    # According to the paper, the minimum of each channel in an area Omega is calculated.
    # In the case of the red channel, the minimum of the inverse is taken.
    min_b = cv2.erode(img[:,:,0], kernel)/255
    min_g = cv2.erode(img[:,:,1], kernel)/255
    min_r = cv2.erode((255-img[:,:,2]), kernel)/255

    b_map = min_b / (1 - bl[0] + 1e-6)
    g_map = min_g / (1 - bl[1] + 1e-6)
    r_map = min_r / (1 - bl[2] + 1e-6)
    
    # At this point a choice needs to be made between clipping and normalizing. The paper does not specify this.
    b_map = cv2.normalize(b_map, None, 0, 1, cv2.NORM_MINMAX)
    g_map = cv2.normalize(g_map, None, 0, 1, cv2.NORM_MINMAX)
    r_map = cv2.normalize(r_map, None, 0, 1, cv2.NORM_MINMAX)

    # b_map = np.clip(b_map, 0, 1)
    # g_map = np.clip(g_map, 0, 1)
    # r_map = np.clip(r_map, 0, 1)
    
    green_red_coeff, blue_red_coeff = calc_coeffs(bl[0], bl[1], bl[2])
    tm_r = 1 - np.minimum(g_map, np.minimum(b_map,r_map))
    tm_g = tm_r ** green_red_coeff
    tm_b = tm_r ** blue_red_coeff

    tm = np.array([tm_b, tm_g, tm_r]).transpose(1,2,0)

    return tm


def get_degraded_img(img:np.ndarray, bl, tm):
    # BGR
    dimg = img.astype(np.float32) / 255
    dimg = dimg * tm + bl * (1 - tm)
    return dimg 
