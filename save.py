import numpy
import cv2
from read_data import convert_Lab2BGR
from matplotlib import pyplot as plt

"""Save images"""
def close_event():
    plt.close()

def saveImages(real_lab_images, generated_lab_images, save_path, columns = 6, display_time = 10000, show = True):

    real_images_lab = real_lab_images.cpu().numpy()
    generated_images_lab = generated_lab_images.cpu().numpy()
    

    fig = plt.figure()
    rows = 3

    for i in range(1, columns+1):

        real_lab = real_images_lab[i]
        generated_lab = generated_images_lab[i]

        real_bgr = convert_Lab2BGR(real_lab)
        real_rgb = cv2.cvtColor(real_bgr,cv2.COLOR_BGR2RGB)
        grayscale = cv2.cvtColor(real_rgb, cv2.COLOR_BGR2GRAY)

        fig.add_subplot(rows, columns, i)
        plt.imshow(grayscale, cmap='gray', vmin = 0, vmax = 255)
        plt.axis('off')
        fig.add_subplot(rows, columns, columns+i)
        plt.imshow(real_rgb)
        plt.axis('off')

        generated_bgr = convert_Lab2BGR(generated_lab)
        generated_rgb = cv2.cvtColor(generated_bgr,cv2.COLOR_BGR2RGB)

        fig.add_subplot(rows, columns, 2*columns+i)
        plt.imshow(generated_rgb)
        plt.axis('off')

    plt.savefig(save_path)
    if show :
        timer = fig.canvas.new_timer(interval = display_time) 
        timer.add_callback(close_event)
        timer.start()
        plt.show()
    else: plt.close()
