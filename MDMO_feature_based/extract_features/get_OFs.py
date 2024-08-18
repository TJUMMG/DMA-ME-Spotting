import dlib
import cv2
import os
import numpy as np
from imutils import face_utils
import scipy.io as sio


p68 = '/landmark_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor_68 = dlib.shape_predictor(p68)


def compute_TVL1(prev, curr):
    """Compute the TV-L1 optical flow."""
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    optical_flow = cv2.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(prev, curr, None)

    assert flow.dtype == np.float32

    return flow


def get_12ROIs(image, detected_face):
    shape68 = predictor_68(image, detected_face)
    shape = face_utils.shape_to_np(shape68)

    ROI_size_half = 8  # CAS(ME)^2

    box_ROI1 = [[shape[18][0] - ROI_size_half, shape[18][1] - ROI_size_half],
                [shape[18][0] + ROI_size_half, shape[18][1] + ROI_size_half]]
    box_ROI2 = [[shape[19][0] - ROI_size_half, shape[19][1] - ROI_size_half],
                [shape[19][0] + ROI_size_half, shape[19][1] + ROI_size_half]]
    box_ROI3 = [[shape[20][0] - ROI_size_half, shape[20][1] - ROI_size_half],
                [shape[20][0] + ROI_size_half, shape[20][1] + ROI_size_half]]
    box_ROI4 = [[shape[23][0] - ROI_size_half, shape[23][1] - ROI_size_half],
                [shape[23][0] + ROI_size_half, shape[23][1] + ROI_size_half]]
    box_ROI5 = [[shape[24][0] - ROI_size_half, shape[24][1] - ROI_size_half],
                [shape[24][0] + ROI_size_half, shape[24][1] + ROI_size_half]]
    box_ROI6 = [[shape[25][0] - ROI_size_half, shape[25][1] - ROI_size_half],
                [shape[25][0] + ROI_size_half, shape[25][1] + ROI_size_half]]
    box_ROI7 = [[shape[30][0] - ROI_size_half, shape[30][1] - ROI_size_half],
                [shape[30][0] + ROI_size_half, shape[30][1] + ROI_size_half]]
    box_ROI8 = [[shape[48][0] - ROI_size_half, shape[48][1] - ROI_size_half],
                [shape[48][0] + ROI_size_half, shape[48][1] + ROI_size_half]]
    box_ROI9 = [[shape[51][0] - ROI_size_half, shape[51][1] - ROI_size_half],
                [shape[51][0] + ROI_size_half, shape[51][1] + ROI_size_half]]
    box_ROI10 = [[shape[54][0] - ROI_size_half, shape[54][1] - ROI_size_half],
                [shape[54][0] + ROI_size_half, shape[54][1] + ROI_size_half]]
    box_ROI11 = [[shape[57][0] - ROI_size_half, shape[57][1] - ROI_size_half],
                [shape[57][0] + ROI_size_half, shape[57][1] + ROI_size_half]]
    box_ROI12 = [[shape[28][0] - ROI_size_half, shape[28][1] - ROI_size_half],
                [shape[28][0] + ROI_size_half, shape[28][1] + ROI_size_half]]

    return box_ROI1, box_ROI2, box_ROI3, box_ROI4, box_ROI5, box_ROI6, box_ROI7, box_ROI8, box_ROI9, box_ROI10, box_ROI11, box_ROI12


def extract_12ROIs_OFs_for_video(video_file):
    save_path ='/results/features/CAS(ME)^2_12ROIs_OFs'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print("Starting !!")
    cap = cv2.VideoCapture(video_file)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Number of frames in this video: ', num_frame)

    idx = 0
    ROI1_frames = []
    ROI2_frames = []
    ROI3_frames = []
    ROI4_frames = []
    ROI5_frames = []
    ROI6_frames = []
    ROI7_frames = []
    ROI8_frames = []
    ROI9_frames = []
    ROI10_frames = []
    ROI11_frames = []
    ROI12_frames = []

    while (cap.isOpened()):
        ret, img = cap.read()
        if (ret == False):
            break
        idx = idx + 1
        if (idx - 1 == 0):
            dets = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2)
            print(dets)
            if (len(dets) > 0):
                iface = 0
                face_distance = 0
                for (j, d) in enumerate(dets):
                    if (d.right() - d.left() > face_distance):
                        face_distance = d.right() - d.left()
                        iface = j
                detected_face = dets[iface]
        box_ROI1, box_ROI2, box_ROI3, box_ROI4, box_ROI5, box_ROI6, box_ROI7, box_ROI8, box_ROI9, box_ROI10, box_ROI11, box_ROI12 = get_12ROIs(img, detected_face)

        ROI1_img = img[box_ROI1[0][1]:box_ROI1[1][1]+1, box_ROI1[0][0]:box_ROI1[1][0]+1]
        ROI2_img = img[box_ROI2[0][1]:box_ROI2[1][1]+1, box_ROI2[0][0]:box_ROI2[1][0]+1]
        ROI3_img = img[box_ROI3[0][1]:box_ROI3[1][1]+1, box_ROI3[0][0]:box_ROI3[1][0]+1]
        ROI4_img = img[box_ROI4[0][1]:box_ROI4[1][1]+1, box_ROI4[0][0]:box_ROI4[1][0]+1]
        ROI5_img = img[box_ROI5[0][1]:box_ROI5[1][1]+1, box_ROI5[0][0]:box_ROI5[1][0]+1]
        ROI6_img = img[box_ROI6[0][1]:box_ROI6[1][1]+1, box_ROI6[0][0]:box_ROI6[1][0]+1]
        ROI7_img = img[box_ROI7[0][1]:box_ROI7[1][1]+1, box_ROI7[0][0]:box_ROI7[1][0]+1]
        ROI8_img = img[box_ROI8[0][1]:box_ROI8[1][1]+1, box_ROI8[0][0]:box_ROI8[1][0]+1]
        ROI9_img = img[box_ROI9[0][1]:box_ROI9[1][1]+1, box_ROI9[0][0]:box_ROI9[1][0]+1]
        ROI10_img = img[box_ROI10[0][1]:box_ROI10[1][1]+1, box_ROI10[0][0]:box_ROI10[1][0]+1]
        ROI11_img = img[box_ROI11[0][1]:box_ROI11[1][1]+1, box_ROI11[0][0]:box_ROI11[1][0]+1]
        ROI12_img = img[box_ROI12[0][1]:box_ROI12[1][1]+1, box_ROI12[0][0]:box_ROI12[1][0]+1]

        ROI1_frames.append(ROI1_img)
        ROI2_frames.append(ROI2_img)
        ROI3_frames.append(ROI3_img)
        ROI4_frames.append(ROI4_img)
        ROI5_frames.append(ROI5_img)
        ROI6_frames.append(ROI6_img)
        ROI7_frames.append(ROI7_img)
        ROI8_frames.append(ROI8_img)
        ROI9_frames.append(ROI9_img)
        ROI10_frames.append(ROI10_img)
        ROI11_frames.append(ROI11_img)
        ROI12_frames.append(ROI12_img)

    cap.release()

    ROIs_frames = [ROI1_frames, ROI2_frames, ROI3_frames, ROI4_frames, ROI5_frames, ROI6_frames,
                   ROI7_frames, ROI8_frames, ROI9_frames, ROI10_frames, ROI11_frames, ROI12_frames]

    ROI1_OF_features = []
    ROI2_OF_features = []
    ROI3_OF_features = []
    ROI4_OF_features = []
    ROI5_OF_features = []
    ROI6_OF_features = []
    ROI7_OF_features = []
    ROI8_OF_features = []
    ROI9_OF_features = []
    ROI10_OF_features = []
    ROI11_OF_features = []
    ROI12_OF_features = []

    for i_frame in range(0, num_frame-1):
        ROI1_OF_features.append(compute_TVL1(ROI1_frames[i_frame], ROI1_frames[i_frame+1]))
        ROI2_OF_features.append(compute_TVL1(ROI2_frames[i_frame], ROI2_frames[i_frame+1]))
        ROI3_OF_features.append(compute_TVL1(ROI3_frames[i_frame], ROI3_frames[i_frame+1]))
        ROI4_OF_features.append(compute_TVL1(ROI4_frames[i_frame], ROI4_frames[i_frame+1]))
        ROI5_OF_features.append(compute_TVL1(ROI5_frames[i_frame], ROI5_frames[i_frame+1]))
        ROI6_OF_features.append(compute_TVL1(ROI6_frames[i_frame], ROI6_frames[i_frame+1]))
        ROI7_OF_features.append(compute_TVL1(ROI7_frames[i_frame], ROI7_frames[i_frame+1]))
        ROI8_OF_features.append(compute_TVL1(ROI8_frames[i_frame], ROI8_frames[i_frame+1]))
        ROI9_OF_features.append(compute_TVL1(ROI9_frames[i_frame], ROI9_frames[i_frame+1]))
        ROI10_OF_features.append(compute_TVL1(ROI10_frames[i_frame], ROI10_frames[i_frame+1]))
        ROI11_OF_features.append(compute_TVL1(ROI11_frames[i_frame], ROI11_frames[i_frame+1]))
        ROI12_OF_features.append(compute_TVL1(ROI12_frames[i_frame], ROI12_frames[i_frame+1]))
        print('Have processed', i_frame, 'frames.')

    ROIs_OF_features = [ROI1_OF_features, ROI2_OF_features, ROI3_OF_features, ROI4_OF_features, ROI5_OF_features, ROI6_OF_features,
                        ROI7_OF_features, ROI8_OF_features, ROI9_OF_features, ROI10_OF_features, ROI11_OF_features, ROI12_OF_features]

    # save results
    save_file = os.path.join(save_path, video_file.split('/')[-1][:7] + '_12ROIs_OF_features.mat')
    sio.savemat(save_file, {'ROIs_frames': ROIs_frames, 'ROIs_OF_features': ROIs_OF_features})

    print('Saving...')


def save_12ROIs_OFs_for_CASme2():
    folder_data = '/dataset/CAS(ME)^2/rawvideo'   # path of CAS(ME)^2 dataset
    subfolders = os.listdir(folder_data)

    for sub_folder in subfolders:
        vidfolders = os.listdir(os.path.join(folder_data, sub_folder))
        for vidname in vidfolders:
            print(vidname)
            extract_12ROIs_OFs_for_video(os.path.join(folder_data, sub_folder, vidname))
    print('Successfully saved.')


def main():
    save_12ROIs_OFs_for_CASme2()


if __name__ == '__main__':
    main()
