import cv2
import os
import argparse
from mtcnn import MTCNN
import time
import warnings
warnings.filterwarnings("ignore")

def isGrayMap(img, threshold = 15):
    img1 = img[:,:,0]
    img2 = img[:,:,1]
    img3 = img[:,:,2]
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    diff_sum = (diff1 + diff2 + diff3) / 3.0
    if diff_sum <= threshold:
        return True
    else:
        return False

def get_data_path(root):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(os.path.sep)[-1], ''), 'detections', i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path

def main(file_dirs):
    detector = MTCNN()
    for file_dir in sorted(os.listdir(file_dirs)):
        file_dir=os.path.join(file_dirs,file_dir)
        save_dir = os.path.join(file_dir,"detections")
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print("mkdir detections")

        im_path, lm_path = get_data_path(file_dir)
        for i in range(len(im_path)):
            print('Detect landmarks:', i, im_path[i])
            image = cv2.imread(im_path[i])

            if isGrayMap(image):
                print('*******detection {} garymap********'.format(im_path[i]))
                continue

            if not os.path.isfile(lm_path[i]):
                detected_faces = detector.detect_faces(image)
                if len(detected_faces) !=0:
                    landmarks=detected_faces[0]['keypoints'].values()

                    # for index, l in enumerate(landmarks):
                    #     pt_pos = (l[0], l[1])
                    #     cv2.circle(image, pt_pos, 1, (0, 225, 0), 5)
                    #     # font = cv2.FONT_HERSHEY_SIMPLEX
                    #     # cv2.putText(img_new, str(index + 1), pt_pos, font, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
                    # cv2.imwrite("output_lm.jpg", image)

                    with open(lm_path[i], "w") as f:
                        for keypoint in landmarks:
                            f.write(str(keypoint[0]) + '\t' + str(keypoint[1]))
                            f.write('\n')
                    f.close()
                else:
                    print('*******detection {} failed********'.format(im_path[i]))
        print('Detect', file_dir, "completed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_folder', required=True, help='folders of training images')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(opt.img_folder)