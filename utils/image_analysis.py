import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
dir_name = ["baking", "croco", "dog", "KAI", "magin", "May"]

for name in dir_name:
    LF_root = "./" + name
    imgs = []
    for (path, dir, files) in os.walk(LF_root):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png' or ext == '.JPG' :
                imgs.append(os.path.join(path,filename))


    for i in range(len(imgs)):
        H, W, C = cv2.imread(imgs[i]).shape
        img = cv2.imread(imgs[i])

        row_axis = list(range(W))

        # 가운데 부분
        R_middle = img[int(H/2), :, 0]
        G_middle = img[int(H/2), :, 1]
        B_middle = img[int(H/2), :, 2]
        # 끝자리 부분
        R_side = img[4, :, 0]
        G_side = img[4, :, 1]
        B_side = img[4, :, 2]

        plt.figure(1)
        plt.plot(row_axis, R_middle.tolist(), color = "red", label="Red")
        plt.plot(row_axis, G_middle.tolist(), color = "green", label="Green")
        plt.plot(row_axis, B_middle.tolist(), color = "blue", label="Blue")
        plt.xlabel("img_row")
        plt.ylabel("pixel_value")
        plt.title("profile line")

        plt.legend()

        plt.figure(2)
        plt.plot(row_axis, R_side.tolist(), color = "red", label="Red")
        plt.plot(row_axis, G_side.tolist(), color = "green", label="Green")
        plt.plot(row_axis, B_side.tolist(), color = "blue", label="Blue")
        plt.xlabel("img_row")
        plt.ylabel("pixel_value")
        plt.title("profile line")
        plt.legend()
        plt.show()
        break # 이미지 한장만 확인
    break # 이미지 한장만 확인
#imgs

#cv2.imread()

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

#img = cv2.imread('99.png', -1)
#cv2.imshow('GoldenGate',img)
#
#color = ('b','g','r')
#for channel,col in enumerate(color):
#    histr = cv2.calcHist([img],[channel],None,[256],[0,256])
#    plt.plot(histr,color = col)
#    plt.xlim([0,256])
#plt.title('Histogram for color scale picture')
#plt.show()
#
#while True:
#    k = cv2.waitKey(0) & 0xFF
#    if k == 27: break             # ESC key to exit
#cv2.destroyAllWindows()

### show fourier
os.getcwd()
LF_root = "./blur_SRFBN"
low_freqs = []
for (path, dir, files) in os.walk(LF_root):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.png' :
            low_freqs.append(os.path.join(path,filename))

path = "./pattern.PNG"
def fourier(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    m_spectrum = 20*np.log(np.abs(fshift))

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title("input image"), plt.xticks([]), plt.yticks([])

    plt.subplot(122), plt.imshow(m_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    print(m_spectrum.shape)
    plt.show()

def compute_fourier(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    m_spectrum = 20*np.log(np.abs(fshift))

    return m_spectrum

def show_fourier(m_spectrum):

    plt.subplot(122), plt.imshow(m_spectrum, cmap='gray')
    plt.title("Magnitude Spectrum"), plt.xticks([]), plt.yticks([])
    plt.show()


m_spectrums = np.zeros((1024, 1024))

for i in range(100):
    m_spectrums += compute_fourier(low_freqs[i])

m_spectrums = m_spectrums/100.
show_fourier(m_spectrums)

#len(low_freqs)
#
#for i in range(25):
#    resize(low_freqs[i])

#
#
#
#
##edge(img_path)
#
#def resize(img_path):
#    img = cv2.imread(img_path)
#    img = cv2.resize(img, dsize=(0,0),fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
#    #img = cv2.resize(img, dsize=(0,0), fx =2, fy = 2, interpolation=cv2.INTER_LINEAR)
#    k = os.path.splitext(img_path)[0][-2:]
#    cv2.imwrite("./prim_0.5/{}.png".format(str(k)), img)
