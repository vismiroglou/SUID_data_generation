import cv2
from matplotlib import pyplot as plt
import numpy as np

from utils import calc_backlight, get_trans_map, get_degraded_img, calc_kernel_size

# Example paper values
B_R = 0.11
B_G = 0.31
B_B = 0.56
T_R = 0.20
T_G = 0.55
T_B = 0.72

if __name__ == '__main__':
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('--input_img', type=str, required=True)
    ap.add_argument('--uw_img', type=str, required=True)
    ap.add_argument('--output_dir', type=str, required=False)
    ap.add_argument('--bl_method', choices=['naive', 'SUID', 'RCP'], default='RCP', type=str, required=False)
    args = ap.parse_args()

    input_img_path = args.input_img
    uw_img_path = args.uw_img

    img = cv2.imread(input_img_path)
    uw_img = cv2.imread(uw_img_path)
    uw_img = cv2.resize(uw_img, (img.shape[1], img.shape[0]))

    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax[0,0].set_title('Input image')
    ax[0,0].set_axis_off()
    ax[0,1].imshow(cv2.cvtColor(uw_img, cv2.COLOR_BGR2RGB))
    ax[0,1].set_title('Underwater image')
    ax[0,1].set_axis_off()

    kernel_size = calc_kernel_size(uw_img)

    bl = calc_backlight(uw_img, method=args.bl_method)
 
    tm = get_trans_map(uw_img, bl, kernel_size=kernel_size)
    ax[1,0].imshow(tm[:,:,2], cmap='gray')
    ax[1,0].set_title('Transmission map')
    ax[1,0].set_axis_off()
    

    dimg = get_degraded_img(img, bl, tm)
    ax[1,1].imshow(cv2.cvtColor((dimg*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    ax[1,1].set_title('Degraded image')
    ax[1,1].set_axis_off()
    fig.tight_layout()
    fig.savefig('process.png') 

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = './'
    
    cv2.imwrite(f'{out_dir}/degraded_image.png', (dimg*255).astype(np.uint8))