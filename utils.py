import cv2
import numpy as np
from matplotlib import pyplot as plt
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

def calc_backlight(img):
    # Calculate the backlight
    # TODO: Do this the proper way. this is an approximation. Backlight is expected to only affect the hue and not the contrast anyway.
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_loc = np.argmax(img_gray)
    max_loc = np.unravel_index(max_loc, img_gray.shape)
    bl = img[max_loc]/255
    # print(bl)
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
    size = (3, 3)
    shape = cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, size)

    min_b = cv2.erode(img[:,:,0], kernel)/255
    min_g = cv2.erode(img[:,:,1], kernel)/255
    min_r = cv2.erode((255-img[:,:,2]), kernel)/255

    # j_red = np.min(np.array([min_b, min_g, 1-min_r]), axis=0)

    b_map = min_b / (1 - bl[0] + 1e-6)
    g_map = min_g / (1 - bl[1] + 1e-6)
    r_map = min_r / (1 - bl[2] + 1e-6)

    print(np.min(b_map), np.min(g_map), np.min(r_map))
    
    # At this point a choice needs to be made between clipping and normalizing. The paper does not specify this.
    # b_map = cv2.normalize(b_map, None, 0, 1, cv2.NORM_MINMAX)
    # g_map = cv2.normalize(g_map, None, 0, 1, cv2.NORM_MINMAX)
    # r_map = cv2.normalize(r_map, None, 0, 1, cv2.NORM_MINMAX)

    b_map = np.clip(b_map, 0, 1)
    g_map = np.clip(g_map, 0, 1)
    r_map = np.clip(r_map, 0, 1)
    
    green_red_coeff, blue_red_coeff = calc_coeffs(bl[0], bl[1], bl[2])
    tm_r = 1 - np.minimum(g_map, np.minimum(b_map,r_map))
    tm_g = tm_r ** green_red_coeff
    tm_b = tm_r ** blue_red_coeff
    # tm_r = np.full(r_map.shape, T_R)
    # tm_g = np.full(g_map.shape, T_G)
    # tm_b = np.full(b_map.shape, T_B)
    
    # fig1, ax1 = plt.subplots(1,3, figsize=(10,3))
    # ax1[0].imshow(tm_b, cmap='gray')
    # ax1[0].set_title('Blue transmission map')
    # ax1[0].set_axis_off()
    # ax1[1].imshow(tm_g, cmap='gray')
    # ax1[1].set_title('Green transmission map')
    # ax1[1].set_axis_off()
    # ax1[2].imshow(tm_r, cmap='gray')
    # ax1[2].set_title('Red transmission map')
    # ax1[2].set_axis_off()
    # fig1.tight_layout()
    # fig1.savefig('transmission_maps.png')

    tm = np.array([tm_b, tm_g, tm_r]).transpose(1,2,0)

    return tm


def get_degraded_img(img:np.ndarray, bl, tm):
    # BGR
    dimg = img.astype(np.float32) / 255
    dimg = dimg * tm + bl * (1 - tm)
    return dimg


if __name__ == '__main__':
    input_img_path = '/home/vismiroglou/datasets/deepBlur/clean/0007.png'
    # input_img_path = '/home/vismiroglou/src/SUD/3008896-1289291431.jpg'
    # uw_img_path = '/home/vismiroglou/datasets/deepBlur/clay/01/0020.png'
    # uw_img_path = '/home/vismiroglou/datasets/brackishMOT/BrackishMOT/train/brackishMOT-01/img1/000001.jpg'
    # uw_img_path = '/home/vismiroglou/datasets/benthicnet/Tasmania201808/r20180822_204918_SS05_beagle_shelf_10/PR_20180822_213745_744_LC16.jpg'
    uw_img_path = '/home/vismiroglou/src/SUD/D328_140_125_0004_600-2230316603.jpg'

    img = cv2.imread(input_img_path)
    uw_img = cv2.imread(uw_img_path)

    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0,0].set_title('Input image')
    ax[0,0].set_axis_off()
    ax[0,1].imshow(cv2.cvtColor(uw_img, cv2.COLOR_BGR2RGB))
    ax[0,1].set_title('Underwater image')
    ax[0,1].set_axis_off()

    bl = calc_backlight(uw_img)
    exit()
    # ax[0,2].imshow(np.tile(np.array([bl[2], bl[1], bl[0]]), (2,2,1)))
    # ax[0,2].set_title('Backlight color')
    # ax[0,2].set_axis_off()
    
 
    tm = get_trans_map(uw_img, bl)
    ax[1,0].imshow(cv2.cvtColor(tm.astype(np.float32), cv2.COLOR_BGR2RGB))
    ax[1,0].set_title('Transmission map')
    ax[1,0].set_axis_off()
    

    img = cv2.resize(img, (uw_img.shape[1], uw_img.shape[0]))
    dimg = get_degraded_img(img, bl, tm)
    ax[1,1].imshow(cv2.cvtColor((dimg*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[1,1].set_title('Degraded image')
    ax[1,1].set_axis_off()
    fig.savefig('figure_5.png')   
