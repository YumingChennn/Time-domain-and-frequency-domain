# import required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt


    
def vissulizeImage(img, dft_shift, mask, dft_shift_and_mask, restored_img, convo_img):
    # visualize input image and the magnitude spectrum
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # first row
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Input Image')
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])

    axs[0, 1].imshow(np.log(1+np.abs(dft_shift)), cmap='gray')
    axs[0, 1].set_title('dft_shift')
    axs[0, 1].set_xticks([])
    axs[0, 1].set_yticks([])

    axs[0, 2].imshow(mask, cmap='gray')
    axs[0, 2].set_title('Mask')
    axs[0, 2].set_xticks([])
    axs[0, 2].set_yticks([])

    # second row
    axs[1, 0].imshow(np.log(1+np.abs(dft_shift_and_mask)), cmap='gray')
    axs[1, 0].set_title('dft_shift_and_mask')
    axs[1, 0].set_xticks([])
    axs[1, 0].set_yticks([])

    axs[1, 1].imshow(np.abs(restored_img), cmap='gray')
    axs[1, 1].set_title('restored_img')
    axs[1, 1].set_xticks([])
    axs[1, 1].set_yticks([])

    axs[1, 2].imshow(np.abs(convo_img), cmap='gray')
    axs[1, 2].set_title('convo_img')
    axs[1, 2].set_xticks([])
    axs[1, 2].set_yticks([])

    plt.show()


def fftFilter(img, kernel):
    rows, cols = img.shape
    extended_kernel = np.zeros((rows,cols), dtype=np.float32)
    extended_kernel[:kernel.shape[0], :kernel.shape[1]] = kernel

    fftkernel = np.fft.fft2(extended_kernel)
    fftkernel_centered = np.abs(np.fft.fftshift(fftkernel))
    return fftkernel_centered

def imageProcess(user_input):
    # read input image
    img = cv2.imread('Lenna.jpg',0)

    #  find the discrete fourier transform of the image and shift zero-frequency component to the center of the spectrum
    dft_shift = np.fft.fftshift(np.fft.fft2(img))

    # Apply mask based on user input
    if user_input == 1:
         # Mean Filter
        kernel = 1/9 * np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])

    elif user_input == 2:
        # Gaussian Filter
        kernel = 1/16 * np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])

    elif user_input == 3:
        # Sobel Filter
        kernel = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
    
    elif user_input == 4:
        # Scharr Filter
        kernel = np.array([[-3, 0, 3],
                            [-10, 0, 10],
                            [-3, 0, 3]])

    elif user_input == 5:
        # Laplacian Filter
        kernel = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    else:
         # Mean Filter
        kernel = 1/9 * np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])
        
    mask = fftFilter(img, kernel)
    dft_shift_and_mask = dft_shift * mask
    # Perform convolution using filter2D
    convo_img = cv2.filter2D(img, -1, kernel)
    
    # Inverse Fourier transform
    restored_img = np.fft.ifft2(np.fft.ifftshift(dft_shift_and_mask))
    
    return img, dft_shift, mask, dft_shift_and_mask, restored_img, convo_img
    

if __name__ == '__main__':
    while True:
        print('Please enter a number from 1 to 5:\n'
        'Entering 1 means applying the mean filter to the photo.\n'
        'Entering 2 means applying the Gaussian filter to the photo.\n'
        'Entering 3 means applying the Sobel filter to the photo.\n'
        'Entering 4 means applying the Scharr filter to the photo.\n'
        'Entering 5 means applying the Laplacian filter to the photo.')
        
        # userinput
        user_input = int(input("What kind of mask you want to apply?"))

        img, dft_shift, mask, dft_shift_and_mask, restored_img, convo_img=imageProcess(user_input)

        vissulizeImage(img, dft_shift, mask, dft_shift_and_mask, restored_img, convo_img)

