import os
import numpy as np
import scipy.io as sio


def get_11ROIs_local_OFs_for_video(video_file):
    print("Starting !!")
    OF_path = r'..\..\..\results\features\CAS(ME)^2_12ROIs_OFs' + '\\'+ video_file.split('\\')[-1][:7] + '_12ROIs_OF_features.mat'

    data = sio.loadmat(OF_path)
    ROIs_OF_features = data['ROIs_OF_features']

    ROI1_OF_features = ROIs_OF_features[0]
    ROI2_OF_features = ROIs_OF_features[1]
    ROI3_OF_features = ROIs_OF_features[2]
    ROI4_OF_features = ROIs_OF_features[3]
    ROI5_OF_features = ROIs_OF_features[4]
    ROI6_OF_features = ROIs_OF_features[5]
    ROI7_OF_features = ROIs_OF_features[6]
    ROI8_OF_features = ROIs_OF_features[7]
    ROI9_OF_features = ROIs_OF_features[8]
    ROI10_OF_features = ROIs_OF_features[9]
    ROI11_OF_features = ROIs_OF_features[10]
    ROI12_OF_features = ROIs_OF_features[11]

    ROI1_local_OF_features = []
    ROI2_local_OF_features = []
    ROI3_local_OF_features = []
    ROI4_local_OF_features = []
    ROI5_local_OF_features = []
    ROI6_local_OF_features = []
    ROI7_local_OF_features = []
    ROI8_local_OF_features = []
    ROI9_local_OF_features = []
    ROI10_local_OF_features = []
    ROI11_local_OF_features = []

    for idx in range(0, len(ROI1_OF_features)):
        ROI12_OF_frame = ROI12_OF_features[idx]
        ROI12_OF_unit = [np.mean(ROI12_OF_frame[:,:,0])/np.sqrt((np.mean(ROI12_OF_frame[:,:,0]))**2 + (np.mean(ROI12_OF_frame[:,:,1]))**2),
                         np.mean(ROI12_OF_frame[:,:,1])/np.sqrt((np.mean(ROI12_OF_frame[:,:,0]))**2 + (np.mean(ROI12_OF_frame[:,:,1]))**2)]
        ROI12_OF_amplitude = np.zeros([ROI12_OF_frame.shape[0], ROI12_OF_frame.shape[1]])
        ROI12_OF_amplitude[:,:] = np.sqrt(ROI12_OF_frame[:,:,0]**2 + ROI12_OF_frame[:,:,1]**2)
        ROI12_OF_average_amplitude = np.mean(ROI12_OF_amplitude)
        ROI12_OF_global_movement = [ROI12_OF_unit[0]*ROI12_OF_average_amplitude, ROI12_OF_unit[1]*ROI12_OF_average_amplitude]

        # ROI1
        ROI1_OF_frame = ROI1_OF_features[idx]
        ROI1_local_OF_frame = np.zeros([ROI1_OF_frame.shape[0], ROI1_OF_frame.shape[1], ROI1_OF_frame.shape[2]])
        ROI1_local_OF_frame[:,:,0] = ROI1_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI1_local_OF_frame[:,:,1] = ROI1_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI1_local_OF_features.append(ROI1_local_OF_frame)

        # ROI2
        ROI2_OF_frame = ROI2_OF_features[idx]
        ROI2_local_OF_frame = np.zeros([ROI2_OF_frame.shape[0], ROI2_OF_frame.shape[1], ROI2_OF_frame.shape[2]])
        ROI2_local_OF_frame[:,:,0] = ROI2_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI2_local_OF_frame[:,:,1] = ROI2_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI2_local_OF_features.append(ROI2_local_OF_frame)

        # ROI3
        ROI3_OF_frame = ROI3_OF_features[idx]
        ROI3_local_OF_frame = np.zeros([ROI3_OF_frame.shape[0], ROI3_OF_frame.shape[1], ROI3_OF_frame.shape[2]])
        ROI3_local_OF_frame[:,:,0] = ROI3_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI3_local_OF_frame[:,:,1] = ROI3_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI3_local_OF_features.append(ROI3_local_OF_frame)

        # ROI4
        ROI4_OF_frame = ROI4_OF_features[idx]
        ROI4_local_OF_frame = np.zeros([ROI4_OF_frame.shape[0], ROI4_OF_frame.shape[1], ROI4_OF_frame.shape[2]])
        ROI4_local_OF_frame[:,:,0] = ROI4_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI4_local_OF_frame[:,:,1] = ROI4_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI4_local_OF_features.append(ROI4_local_OF_frame)

        # ROI5
        ROI5_OF_frame = ROI5_OF_features[idx]
        ROI5_local_OF_frame = np.zeros([ROI5_OF_frame.shape[0], ROI5_OF_frame.shape[1], ROI5_OF_frame.shape[2]])
        ROI5_local_OF_frame[:,:,0] = ROI5_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI5_local_OF_frame[:,:,1] = ROI5_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI5_local_OF_features.append(ROI5_local_OF_frame)

        # ROI6
        ROI6_OF_frame = ROI6_OF_features[idx]
        ROI6_local_OF_frame = np.zeros([ROI6_OF_frame.shape[0], ROI6_OF_frame.shape[1], ROI6_OF_frame.shape[2]])
        ROI6_local_OF_frame[:,:,0] = ROI6_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI6_local_OF_frame[:,:,1] = ROI6_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI6_local_OF_features.append(ROI6_local_OF_frame)

        # ROI7
        ROI7_OF_frame = ROI7_OF_features[idx]
        ROI7_local_OF_frame = np.zeros([ROI7_OF_frame.shape[0], ROI7_OF_frame.shape[1], ROI7_OF_frame.shape[2]])
        ROI7_local_OF_frame[:,:,0] = ROI7_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI7_local_OF_frame[:,:,1] = ROI7_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI7_local_OF_features.append(ROI7_local_OF_frame)


        # ROI8
        ROI8_OF_frame = ROI8_OF_features[idx]
        ROI8_local_OF_frame = np.zeros([ROI8_OF_frame.shape[0], ROI8_OF_frame.shape[1], ROI8_OF_frame.shape[2]])
        ROI8_local_OF_frame[:,:,0] = ROI8_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI8_local_OF_frame[:,:,1] = ROI8_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI8_local_OF_features.append(ROI8_local_OF_frame)

        # ROI9
        ROI9_OF_frame = ROI9_OF_features[idx]
        ROI9_local_OF_frame = np.zeros([ROI9_OF_frame.shape[0], ROI9_OF_frame.shape[1], ROI9_OF_frame.shape[2]])
        ROI9_local_OF_frame[:,:,0] = ROI9_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI9_local_OF_frame[:,:,1] = ROI9_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI9_local_OF_features.append(ROI9_local_OF_frame)

        # ROI10
        ROI10_OF_frame = ROI10_OF_features[idx]
        ROI10_local_OF_frame = np.zeros([ROI10_OF_frame.shape[0], ROI10_OF_frame.shape[1], ROI10_OF_frame.shape[2]])
        ROI10_local_OF_frame[:,:,0] = ROI10_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI10_local_OF_frame[:,:,1] = ROI10_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI10_local_OF_features.append(ROI10_local_OF_frame)


        # ROI11
        ROI11_OF_frame = ROI11_OF_features[idx]
        ROI11_local_OF_frame = np.zeros([ROI11_OF_frame.shape[0], ROI11_OF_frame.shape[1], ROI11_OF_frame.shape[2]])
        ROI11_local_OF_frame[:,:,0] = ROI11_OF_frame[:,:,0] - ROI12_OF_global_movement[0]
        ROI11_local_OF_frame[:,:,1] = ROI11_OF_frame[:,:,1] - ROI12_OF_global_movement[1]
        ROI11_local_OF_features.append(ROI11_local_OF_frame)

        print('Have processed', idx, 'frames.')

    ROIs_local_OF_features = [ROI1_local_OF_features, ROI2_local_OF_features, ROI3_local_OF_features, ROI4_local_OF_features, ROI5_local_OF_features, ROI6_local_OF_features,
                              ROI7_local_OF_features, ROI8_local_OF_features, ROI9_local_OF_features, ROI10_local_OF_features, ROI11_local_OF_features]
    return ROIs_local_OF_features


def save_11ROIs_local_OFs_for_CASme2():
    save_path = r'..\..\..\results\features\CAS(ME)^2_11ROIs_local_OFs'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    folder_data = r'D:\ME_Database_Download\CAS(ME)^2\rawvideo'   # path of CAS(ME)^2 dataset
    subfolders = os.listdir(folder_data)

    for sub_folder in subfolders:
        vidfolders = os.listdir(os.path.join(folder_data, sub_folder))
        for vidname in vidfolders:
            print(vidname)
            ROIs_local_OF_features = get_11ROIs_local_OFs_for_video(os.path.join(folder_data, sub_folder, vidname))

            # save results
            save_file = os.path.join(save_path, vidname[:7] + '_11ROIs_local_OF_features.mat')
            sio.savemat(save_file, {'ROIs_local_OF_features': ROIs_local_OF_features})
            print('Saving...')

    print('Successfully saved.')


def main():
    save_11ROIs_local_OFs_for_CASme2()


if __name__ == '__main__':
    main()
