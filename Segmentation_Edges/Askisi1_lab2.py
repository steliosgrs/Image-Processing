import cv2
import numpy as np

from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score,davies_bouldin_score
from matplotlib import colors
from skimage.color import label2rgb

def plot3d(x,y,z):
    fig = plt.figure()
    axis = fig.add_subplot(1,1,1,projection ="3d")
    axis.scatter(x.flatten(), y.flatten(), z.flatten(), facecolors=pixel_colors, marker=".")
    # axis.scatter(x, y, z, marker='o', facecolors=cv2.cvtColor(img, cv2.COLOR_BGR2RGB).reshape(-1,3)/255.)
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.show()

def plotHSV(h,s,v):
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

def createHist(anImage):
    imgNormHist = cv2.calcHist([anImage], [0], None, [256], [0, 256])
    imgNormHist = imgNormHist / sum(imgNormHist)

    return imgNormHist

#define a performance function
# we need this to evaluate (somehow) how well the clustering is done
def performance_score(input_values, cluster_indexes):
    try:
        silh_score = silhouette_score(input_values, cluster_indexes)
        print(' .. Silhouette Coefficient score is {:.2f}'.format(silh_score))
        print( ' ... -1: incorrect, 0: overlapping, +1: highly dense clusters.')
    except:
        print(' .. Warning: could not calculate Silhouette Coefficient score.')
        silh_score = -999

    try:
        ch_score = calinski_harabasz_score(input_values, cluster_indexes)
        print(' .. Calinski-Harabasz Index score is {:.2f}'.format(ch_score))
        print(' ... Higher the value better the clusters.')
    except:
        print(' .. Warning: could not calculate Calinski-Harabasz Index score.')
        ch_score = -999

    try:
        db_score = davies_bouldin_score(input_values, cluster_indexes)
        print(' .. Davies-Bouldin Index score is {:.2f}'.format(db_score))
        print(' ... 0: Lowest possible value, good partitioning.')
    except:
        print(' .. Warning: could not calculate Davies-Bouldin Index Index score.')
        db_score = -999

    return silh_score, ch_score, db_score

def segmentationMS(img,type_img):
    # Shape of original image
    originShape = img.shape
    # reshape the image in appropriate data format to use for clustering
    flat_img = np.reshape(img, [-1, 3])

    # MeanShift calcualtes the number of clusters
    # Estimate bandwidth for meanshift algorithm
    bandwidth = estimate_bandwidth(flat_img, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(flat_img)
    # (r,g,b) vectors corresponding to the different clusters after meanshift
    labels=ms.labels_

    uniqueLabels = np.unique(labels)
    print('we have a total of {:d} unique clusters'.format(len(uniqueLabels)))

    # evaluate the performance
    # _,_,_ = performance_score(flat_img, labels)

    simg = np.reshape(labels,originShape[:2])
    # Θα πρέπει να χρησιμοποιίσουμε την συνάρτηση label2rgb
    # για να μην εμφανιστούν μαύρα τα labels της εικόνας και να
    # λειτουργήσει σωστά το cv2.imshow.
    # Σωστή κωδικοποίηση των δεδομένων της εικόνας με το astype(np.uint8)
    simg =( label2rgb(simg, bg_label=0) * 255).astype(np.uint8)

    # display and save Segmented Image
    cv2.namedWindow("Segmented Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Segmented Image", simg)
    # cv2.imwrite(f"MeanShift Segmented {type_img} Image.png",simg)
    cv2.waitKey()
    return simg

def segmentationKM(img,type_img):
    # Shape of original image
    originShape = img.shape
    # reshape the image in appropriate data format to use for clustering
    flat_img = np.reshape(img, [-1, 3])
    # now go for the kmeans
    km = MiniBatchKMeans(n_clusters=8)
    km.fit(flat_img)
    labels = km.labels_
    uniqueLabels = np.unique(labels)
    print('KMEANS we have a total of {:d} unique clusters'.format(len(uniqueLabels)))

    # evaluate the performance
    # _, _, _ = performance_score(flat_img, km.labels_)

    # Θα πρέπει να χρησιμοποιίσουμε την συνάρτηση label2rgb
    # για να μην εμφανιστούν μαύρα τα labels της εικόνας και να
    # λειτουργήσει σωστά το cv2.imshow.
    # Σωστή κωδικοποίηση των δεδομένων της εικόνας με το astype(np.uint8)
    segmentedImgKM = np.reshape(labels, originShape[:2])
    segmentedImgKM = label2rgb(segmentedImgKM, bg_label=0) * 255  # need this to work with cv2. imshow

    # display and save Segmented Image
    cv2.namedWindow("kmeansSegments", cv2.WINDOW_NORMAL)
    cv2.imshow("kmeansSegments", segmentedImgKM.astype(np.uint8))
    # cv2.imwrite(f"KMeans - Segmented {type_img}  Image.png", segmentedImgKM)
    cv2.waitKey()
    return segmentedImgKM


if __name__ == '__main__':

    # read the image and convert image to RGB
    img = cv2.imread('hohenheim.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()

    # Split the channels to R G B
    red_ch, green_ch, blue_ch = cv2.split(img)

    # Normalization
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    # 3D Plot RGB τιμών
    # plot3d(red_ch, green_ch, blue_ch)

    # Convert and Split the channels to H S V
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)

    # 3D Plot HSV τιμών
    # plotHSV(h, s, v)

    # Channels RGB & HVS
    # blue_Histogram = createHist(blue_ch)
    # green_Histogram = createHist(green_ch)
    # red_Histogram = createHist(red_ch)


    # Histogram RGB

    # plt.title("Histogram RGB")
    # plt.plot(blue_Histogram, label="Blue",color='blue')
    # plt.plot(green_Histogram, label="Green", color='green')
    # plt.plot(red_Histogram,label="Red", color='red')
    # plt.legend(loc='best')
    # plt.show()

    # Histogram HSV
    # h_Histogram = createHist(h)
    # s_Histogram = createHist(s)
    # v_Histogram = createHist(v)
    # plt.title("Histogram HSV")
    # plt.plot(h_Histogram, label="Hue",color='orange')
    # plt.plot(s_Histogram, label="Saturation", color='green')
    # plt.plot(v_Histogram,label="Value", color='purple')
    # plt.legend(loc='best')
    # plt.show()

    # Segmentation
    img = cv2.imread('hohenheim.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # MeanShift
    imagetype = 'RGB'
    ms_rgb_segm_img = segmentationMS(img,imagetype)
    imagetype = 'HSV'
    ms_hsv_segm_img = segmentationMS(hsv_img,imagetype)

    # KMeans
    imagetype = 'RGB'
    km_rgb_segm_img = segmentationKM(img,imagetype)
    imagetype = 'HSV'
    km_hsv_segm_img = segmentationKM(hsv_img,imagetype)


    # HSV + RGB
    # Merged Segmented Images
    ms_merged = ms_rgb_segm_img + ms_hsv_segm_img
    km_merged = km_rgb_segm_img + km_hsv_segm_img

    # display and save Merged Images
    # cv2.namedWindow("MS_Merged", cv2.WINDOW_NORMAL)
    # cv2.imshow("MS_Merged", ms_merged)
    # cv2.imwrite("MeanShift - Merged Image.png", ms_merged)
    # cv2.waitKey()
    # cv2.namedWindow("kmeansMerged", cv2.WINDOW_NORMAL)
    # cv2.imshow("kmeansMerged", km_merged)
    # cv2.imwrite("KMeans - Merged Image.png", km_merged)
    # cv2.imwrite("KMeans - Merged Image.png", km_merged)
    # cv2.waitKey()

    # Clear All windows
    cv2.destroyAllWindows()