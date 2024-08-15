import os
import numpy as np
import math
import xlwt
import scipy.io as sio
import cmath


def get_MDMO_feature_excel_for_video(video_file):
    print("Starting !!")
    local_OF_path = r'..\..\..\results\features\CAS(ME)^2_11ROIs_local_OFs' + '\\' + video_file.split('\\')[-1][:7] + '_11ROIs_local_OF_features.mat'

    data = sio.loadmat(local_OF_path)
    ROIs_local_OF_features = data['ROIs_local_OF_features']

    ROI1_local_OF_features = ROIs_local_OF_features[0]
    ROI2_local_OF_features = ROIs_local_OF_features[1]
    ROI3_local_OF_features = ROIs_local_OF_features[2]
    ROI4_local_OF_features = ROIs_local_OF_features[3]
    ROI5_local_OF_features = ROIs_local_OF_features[4]
    ROI6_local_OF_features = ROIs_local_OF_features[5]
    ROI7_local_OF_features = ROIs_local_OF_features[6]
    ROI8_local_OF_features = ROIs_local_OF_features[7]
    ROI9_local_OF_features = ROIs_local_OF_features[8]
    ROI10_local_OF_features = ROIs_local_OF_features[9]
    ROI11_local_OF_features = ROIs_local_OF_features[10]

    ROI1_MDMO_rhos = []
    ROI1_MDMO_angles = []
    ROI2_MDMO_rhos = []
    ROI2_MDMO_angles = []
    ROI3_MDMO_rhos = []
    ROI3_MDMO_angles = []
    ROI4_MDMO_rhos = []
    ROI4_MDMO_angles = []
    ROI5_MDMO_rhos = []
    ROI5_MDMO_angles = []
    ROI6_MDMO_rhos = []
    ROI6_MDMO_angles = []
    ROI7_MDMO_rhos = []
    ROI7_MDMO_angles = []
    ROI8_MDMO_rhos = []
    ROI8_MDMO_angles = []
    ROI9_MDMO_rhos = []
    ROI9_MDMO_angles = []
    ROI10_MDMO_rhos = []
    ROI10_MDMO_angles = []
    ROI11_MDMO_rhos = []
    ROI11_MDMO_angles = []
    for idx in range(0, len(ROI1_local_OF_features)):
        # ROI1
        ROI1_local_OF_frame = ROI1_local_OF_features[idx]
        ROI1_local_OF_polar_frame = np.zeros([ROI1_local_OF_frame.shape[0], ROI1_local_OF_frame.shape[1], ROI1_local_OF_frame.shape[2]])
        for i in range(0, ROI1_local_OF_frame.shape[0]):
            for j in range(0, ROI1_local_OF_frame.shape[1]):
                ROI1_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI1_local_OF_frame[i, j, 0], ROI1_local_OF_frame[i, j, 1]))[0]
                ROI1_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI1_local_OF_frame[i, j, 0], ROI1_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI1_local_OF_frame.shape[0]):
            for j in range(0, ROI1_local_OF_frame.shape[1]):
                if (ROI1_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI1_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI1_local_OF_polar_frame[i, j, 0]
                if (ROI1_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI1_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI1_local_OF_polar_frame[i, j, 0]
                if ((ROI1_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI1_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI1_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI1_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI1_local_OF_polar_frame[i, j, 0]
                if (ROI1_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI1_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI1_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI1_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI1_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI1_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI1_MDMO_rhos.append(cmath.polar(complex(ROI1_main_vector[0], ROI1_main_vector[1]))[0])
        ROI1_MDMO_angles.append(cmath.polar(complex(ROI1_main_vector[0], ROI1_main_vector[1]))[1])


        # ROI2
        ROI2_local_OF_frame = ROI2_local_OF_features[idx]
        ROI2_local_OF_polar_frame = np.zeros([ROI2_local_OF_frame.shape[0], ROI2_local_OF_frame.shape[1], ROI2_local_OF_frame.shape[2]])
        for i in range(0, ROI2_local_OF_frame.shape[0]):
            for j in range(0, ROI2_local_OF_frame.shape[1]):
                ROI2_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI2_local_OF_frame[i, j, 0], ROI2_local_OF_frame[i, j, 1]))[0]
                ROI2_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI2_local_OF_frame[i, j, 0], ROI2_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI2_local_OF_frame.shape[0]):
            for j in range(0, ROI2_local_OF_frame.shape[1]):
                if (ROI2_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI2_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI2_local_OF_polar_frame[i, j, 0]
                if (ROI2_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI2_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI2_local_OF_polar_frame[i, j, 0]
                if ((ROI2_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI2_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI2_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI2_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI2_local_OF_polar_frame[i, j, 0]
                if (ROI2_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI2_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI2_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI2_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI2_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI2_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI2_MDMO_rhos.append(cmath.polar(complex(ROI2_main_vector[0], ROI2_main_vector[1]))[0])
        ROI2_MDMO_angles.append(cmath.polar(complex(ROI2_main_vector[0], ROI2_main_vector[1]))[1])


        # ROI3
        ROI3_local_OF_frame = ROI3_local_OF_features[idx]
        ROI3_local_OF_polar_frame = np.zeros([ROI3_local_OF_frame.shape[0], ROI3_local_OF_frame.shape[1], ROI3_local_OF_frame.shape[2]])
        for i in range(0, ROI3_local_OF_frame.shape[0]):
            for j in range(0, ROI3_local_OF_frame.shape[1]):
                ROI3_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI3_local_OF_frame[i, j, 0], ROI3_local_OF_frame[i, j, 1]))[0]
                ROI3_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI3_local_OF_frame[i, j, 0], ROI3_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI3_local_OF_frame.shape[0]):
            for j in range(0, ROI3_local_OF_frame.shape[1]):
                if (ROI3_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI3_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI3_local_OF_polar_frame[i, j, 0]
                if (ROI3_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI3_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI3_local_OF_polar_frame[i, j, 0]
                if ((ROI3_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI3_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI3_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI3_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI3_local_OF_polar_frame[i, j, 0]
                if (ROI3_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI3_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI3_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI3_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI3_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI3_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI3_MDMO_rhos.append(cmath.polar(complex(ROI3_main_vector[0], ROI3_main_vector[1]))[0])
        ROI3_MDMO_angles.append(cmath.polar(complex(ROI3_main_vector[0], ROI3_main_vector[1]))[1])


        # ROI4
        ROI4_local_OF_frame = ROI4_local_OF_features[idx]
        ROI4_local_OF_polar_frame = np.zeros([ROI4_local_OF_frame.shape[0], ROI4_local_OF_frame.shape[1], ROI4_local_OF_frame.shape[2]])
        for i in range(0, ROI4_local_OF_frame.shape[0]):
            for j in range(0, ROI4_local_OF_frame.shape[1]):
                ROI4_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI4_local_OF_frame[i, j, 0], ROI4_local_OF_frame[i, j, 1]))[0]
                ROI4_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI4_local_OF_frame[i, j, 0], ROI4_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI4_local_OF_frame.shape[0]):
            for j in range(0, ROI4_local_OF_frame.shape[1]):
                if (ROI4_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI4_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI4_local_OF_polar_frame[i, j, 0]
                if (ROI4_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI4_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI4_local_OF_polar_frame[i, j, 0]
                if ((ROI4_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI4_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI4_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI4_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI4_local_OF_polar_frame[i, j, 0]
                if (ROI4_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI4_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI4_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI4_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI4_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI4_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI4_MDMO_rhos.append(cmath.polar(complex(ROI4_main_vector[0], ROI4_main_vector[1]))[0])
        ROI4_MDMO_angles.append(cmath.polar(complex(ROI4_main_vector[0], ROI4_main_vector[1]))[1])


        # ROI5
        ROI5_local_OF_frame = ROI5_local_OF_features[idx]
        ROI5_local_OF_polar_frame = np.zeros([ROI5_local_OF_frame.shape[0], ROI5_local_OF_frame.shape[1], ROI5_local_OF_frame.shape[2]])
        for i in range(0, ROI5_local_OF_frame.shape[0]):
            for j in range(0, ROI5_local_OF_frame.shape[1]):
                ROI5_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI5_local_OF_frame[i, j, 0], ROI5_local_OF_frame[i, j, 1]))[0]
                ROI5_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI5_local_OF_frame[i, j, 0], ROI5_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI5_local_OF_frame.shape[0]):
            for j in range(0, ROI5_local_OF_frame.shape[1]):
                if (ROI5_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI5_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI5_local_OF_polar_frame[i, j, 0]
                if (ROI5_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI5_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI5_local_OF_polar_frame[i, j, 0]
                if ((ROI5_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI5_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI5_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI5_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI5_local_OF_polar_frame[i, j, 0]
                if (ROI5_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI5_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI5_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI5_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI5_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI5_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI5_MDMO_rhos.append(cmath.polar(complex(ROI5_main_vector[0], ROI5_main_vector[1]))[0])
        ROI5_MDMO_angles.append(cmath.polar(complex(ROI5_main_vector[0], ROI5_main_vector[1]))[1])


        # ROI6
        ROI6_local_OF_frame = ROI6_local_OF_features[idx]
        ROI6_local_OF_polar_frame = np.zeros([ROI6_local_OF_frame.shape[0], ROI6_local_OF_frame.shape[1], ROI6_local_OF_frame.shape[2]])
        for i in range(0, ROI6_local_OF_frame.shape[0]):
            for j in range(0, ROI6_local_OF_frame.shape[1]):
                ROI6_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI6_local_OF_frame[i, j, 0], ROI6_local_OF_frame[i, j, 1]))[0]
                ROI6_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI6_local_OF_frame[i, j, 0], ROI6_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI6_local_OF_frame.shape[0]):
            for j in range(0, ROI6_local_OF_frame.shape[1]):
                if (ROI6_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI6_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI6_local_OF_polar_frame[i, j, 0]
                if (ROI6_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI6_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI6_local_OF_polar_frame[i, j, 0]
                if ((ROI6_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI6_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI6_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI6_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI6_local_OF_polar_frame[i, j, 0]
                if (ROI6_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI6_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI6_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI6_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI6_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI6_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI6_MDMO_rhos.append(cmath.polar(complex(ROI6_main_vector[0], ROI6_main_vector[1]))[0])
        ROI6_MDMO_angles.append(cmath.polar(complex(ROI6_main_vector[0], ROI6_main_vector[1]))[1])


        # ROI7
        ROI7_local_OF_frame = ROI7_local_OF_features[idx]
        ROI7_local_OF_polar_frame = np.zeros([ROI7_local_OF_frame.shape[0], ROI7_local_OF_frame.shape[1], ROI7_local_OF_frame.shape[2]])
        for i in range(0, ROI7_local_OF_frame.shape[0]):
            for j in range(0, ROI7_local_OF_frame.shape[1]):
                ROI7_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI7_local_OF_frame[i, j, 0], ROI7_local_OF_frame[i, j, 1]))[0]
                ROI7_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI7_local_OF_frame[i, j, 0], ROI7_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI7_local_OF_frame.shape[0]):
            for j in range(0, ROI7_local_OF_frame.shape[1]):
                if (ROI7_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI7_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI7_local_OF_polar_frame[i, j, 0]
                if (ROI7_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI7_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI7_local_OF_polar_frame[i, j, 0]
                if ((ROI7_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI7_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI7_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI7_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI7_local_OF_polar_frame[i, j, 0]
                if (ROI7_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI7_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI7_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI7_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI7_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI7_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI7_MDMO_rhos.append(cmath.polar(complex(ROI7_main_vector[0], ROI7_main_vector[1]))[0])
        ROI7_MDMO_angles.append(cmath.polar(complex(ROI7_main_vector[0], ROI7_main_vector[1]))[1])


        # ROI8
        ROI8_local_OF_frame = ROI8_local_OF_features[idx]
        ROI8_local_OF_polar_frame = np.zeros([ROI8_local_OF_frame.shape[0], ROI8_local_OF_frame.shape[1], ROI8_local_OF_frame.shape[2]])
        for i in range(0, ROI8_local_OF_frame.shape[0]):
            for j in range(0, ROI8_local_OF_frame.shape[1]):
                ROI8_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI8_local_OF_frame[i, j, 0], ROI8_local_OF_frame[i, j, 1]))[0]
                ROI8_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI8_local_OF_frame[i, j, 0], ROI8_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI8_local_OF_frame.shape[0]):
            for j in range(0, ROI8_local_OF_frame.shape[1]):
                if (ROI8_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI8_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI8_local_OF_polar_frame[i, j, 0]
                if (ROI8_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI8_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI8_local_OF_polar_frame[i, j, 0]
                if ((ROI8_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI8_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI8_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI8_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI8_local_OF_polar_frame[i, j, 0]
                if (ROI8_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI8_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI8_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI8_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI8_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI8_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI8_MDMO_rhos.append(cmath.polar(complex(ROI8_main_vector[0], ROI8_main_vector[1]))[0])
        ROI8_MDMO_angles.append(cmath.polar(complex(ROI8_main_vector[0], ROI8_main_vector[1]))[1])


        # ROI9
        ROI9_local_OF_frame = ROI9_local_OF_features[idx]
        ROI9_local_OF_polar_frame = np.zeros([ROI9_local_OF_frame.shape[0], ROI9_local_OF_frame.shape[1], ROI9_local_OF_frame.shape[2]])
        for i in range(0, ROI9_local_OF_frame.shape[0]):
            for j in range(0, ROI9_local_OF_frame.shape[1]):
                ROI9_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI9_local_OF_frame[i, j, 0], ROI9_local_OF_frame[i, j, 1]))[0]
                ROI9_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI9_local_OF_frame[i, j, 0], ROI9_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI9_local_OF_frame.shape[0]):
            for j in range(0, ROI9_local_OF_frame.shape[1]):
                if (ROI9_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI9_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI9_local_OF_polar_frame[i, j, 0]
                if (ROI9_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI9_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI9_local_OF_polar_frame[i, j, 0]
                if ((ROI9_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI9_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI9_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI9_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI9_local_OF_polar_frame[i, j, 0]
                if (ROI9_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI9_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI9_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI9_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI9_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI9_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI9_MDMO_rhos.append(cmath.polar(complex(ROI9_main_vector[0], ROI9_main_vector[1]))[0])
        ROI9_MDMO_angles.append(cmath.polar(complex(ROI9_main_vector[0], ROI9_main_vector[1]))[1])


        # ROI10
        ROI10_local_OF_frame = ROI10_local_OF_features[idx]
        ROI10_local_OF_polar_frame = np.zeros([ROI10_local_OF_frame.shape[0], ROI10_local_OF_frame.shape[1], ROI10_local_OF_frame.shape[2]])
        for i in range(0, ROI10_local_OF_frame.shape[0]):
            for j in range(0, ROI10_local_OF_frame.shape[1]):
                ROI10_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI10_local_OF_frame[i, j, 0], ROI10_local_OF_frame[i, j, 1]))[0]
                ROI10_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI10_local_OF_frame[i, j, 0], ROI10_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI10_local_OF_frame.shape[0]):
            for j in range(0, ROI10_local_OF_frame.shape[1]):
                if (ROI10_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI10_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI10_local_OF_polar_frame[i, j, 0]
                if (ROI10_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI10_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI10_local_OF_polar_frame[i, j, 0]
                if ((ROI10_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI10_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI10_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI10_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI10_local_OF_polar_frame[i, j, 0]
                if (ROI10_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI10_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI10_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI10_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI10_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI10_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI10_MDMO_rhos.append(cmath.polar(complex(ROI10_main_vector[0], ROI10_main_vector[1]))[0])
        ROI10_MDMO_angles.append(cmath.polar(complex(ROI10_main_vector[0], ROI10_main_vector[1]))[1])


        # ROI11
        ROI11_local_OF_frame = ROI11_local_OF_features[idx]
        ROI11_local_OF_polar_frame = np.zeros([ROI11_local_OF_frame.shape[0], ROI11_local_OF_frame.shape[1], ROI11_local_OF_frame.shape[2]])
        for i in range(0, ROI11_local_OF_frame.shape[0]):
            for j in range(0, ROI11_local_OF_frame.shape[1]):
                ROI11_local_OF_polar_frame[i, j, 0] = cmath.polar(complex(ROI11_local_OF_frame[i, j, 0], ROI11_local_OF_frame[i, j, 1]))[0]
                ROI11_local_OF_polar_frame[i, j, 1] = cmath.polar(complex(ROI11_local_OF_frame[i, j, 0], ROI11_local_OF_frame[i, j, 1]))[1]

        orientation1_positions = []
        orientation2_positions = []
        orientation3_positions = []
        orientation4_positions = []
        orientation1_magnitude_sum = 0.0
        orientation2_magnitude_sum = 0.0
        orientation3_magnitude_sum = 0.0
        orientation4_magnitude_sum = 0.0
        for i in range(0, ROI11_local_OF_frame.shape[0]):
            for j in range(0, ROI11_local_OF_frame.shape[1]):
                if (ROI11_local_OF_polar_frame[i,j,1] >= -math.pi/6) and (ROI11_local_OF_polar_frame[i,j,1] < math.pi/6):
                    orientation1_positions.append([i,j])
                    orientation1_magnitude_sum += ROI11_local_OF_polar_frame[i, j, 0]
                if (ROI11_local_OF_polar_frame[i,j,1] >= math.pi/6) and (ROI11_local_OF_polar_frame[i,j,1] < math.pi*5/6):
                    orientation2_positions.append([i,j])
                    orientation2_magnitude_sum += ROI11_local_OF_polar_frame[i, j, 0]
                if ((ROI11_local_OF_polar_frame[i,j,1] >= math.pi*5/6) and (ROI11_local_OF_polar_frame[i,j,1] <= math.pi)) or ((ROI11_local_OF_polar_frame[i,j,1] > -math.pi) and (ROI11_local_OF_polar_frame[i,j,1] <= -math.pi*5/6)):
                    orientation3_positions.append([i,j])
                    orientation3_magnitude_sum += ROI11_local_OF_polar_frame[i, j, 0]
                if (ROI11_local_OF_polar_frame[i,j,1] >= -math.pi*5/6) and (ROI11_local_OF_polar_frame[i,j,1] < -math.pi/6):
                    orientation4_positions.append([i,j])
                    orientation4_magnitude_sum += ROI11_local_OF_polar_frame[i, j, 0]

        orientations_positions = [orientation1_positions, orientation2_positions, orientation3_positions, orientation4_positions]
        orientations_magnitude_sums = [orientation1_magnitude_sum, orientation2_magnitude_sum, orientation3_magnitude_sum, orientation4_magnitude_sum]
        max_idx = [index for index in range(0, len(orientations_magnitude_sums)) if orientations_magnitude_sums[index] == max(orientations_magnitude_sums)]
        main_orientation_positions = orientations_positions[max_idx[0]]
        print(len(main_orientation_positions))
        main_orientation_magnitude_average = orientations_magnitude_sums[max_idx[0]] / len(main_orientation_positions)

        main_orientation_vector_sum = [0.0, 0.0]
        for i_position in range(0, len(main_orientation_positions)):
            main_orientation_vector_sum[0] += ROI11_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 0]
            main_orientation_vector_sum[1] += ROI11_local_OF_frame[main_orientation_positions[i_position][0], main_orientation_positions[i_position][1], 1]

        main_orientation_unit_vector = [main_orientation_vector_sum[0]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2),
                                        main_orientation_vector_sum[1]/np.sqrt(main_orientation_vector_sum[0]**2+main_orientation_vector_sum[1]**2)]
        ROI11_main_vector = [main_orientation_unit_vector[0]*main_orientation_magnitude_average, main_orientation_unit_vector[1]*main_orientation_magnitude_average]
        ROI11_MDMO_rhos.append(cmath.polar(complex(ROI11_main_vector[0], ROI11_main_vector[1]))[0])
        ROI11_MDMO_angles.append(cmath.polar(complex(ROI11_main_vector[0], ROI11_main_vector[1]))[1])

    return ROI1_MDMO_rhos, ROI1_MDMO_angles, ROI2_MDMO_rhos, ROI2_MDMO_angles, ROI3_MDMO_rhos, ROI3_MDMO_angles, ROI4_MDMO_rhos, ROI4_MDMO_angles, ROI5_MDMO_rhos, ROI5_MDMO_angles, ROI6_MDMO_rhos, ROI6_MDMO_angles, ROI7_MDMO_rhos, ROI7_MDMO_angles, ROI8_MDMO_rhos, ROI8_MDMO_angles, ROI9_MDMO_rhos, ROI9_MDMO_angles, ROI10_MDMO_rhos, ROI10_MDMO_angles, ROI11_MDMO_rhos, ROI11_MDMO_angles


def save_MDMO_features_for_CASme2():
    rst_path = r'..\..\..\results\features\CAS(ME)^2_11ROIs_MDMO_features'
    if not os.path.exists(rst_path):
        os.mkdir(rst_path)
    folder_data = r'D:\ME_Database_Download\CAS(ME)^2\rawvideo'   # path of CAS(ME)^2 dataset
    subfolders = os.listdir(folder_data)

    for sub_folder in subfolders:
        vidfolders = os.listdir(os.path.join(folder_data, sub_folder))
        for vidname in vidfolders:
            print(vidname)
            ROI1_MDMO_rhos, ROI1_MDMO_angles, ROI2_MDMO_rhos, ROI2_MDMO_angles, ROI3_MDMO_rhos, ROI3_MDMO_angles, ROI4_MDMO_rhos, ROI4_MDMO_angles, ROI5_MDMO_rhos, ROI5_MDMO_angles, ROI6_MDMO_rhos, ROI6_MDMO_angles, ROI7_MDMO_rhos, ROI7_MDMO_angles, ROI8_MDMO_rhos, ROI8_MDMO_angles, ROI9_MDMO_rhos, ROI9_MDMO_angles, ROI10_MDMO_rhos, ROI10_MDMO_angles, ROI11_MDMO_rhos, ROI11_MDMO_angles = get_MDMO_feature_excel_for_video(os.path.join(folder_data, sub_folder, vidname))

            # save results
            if not os.path.exists(os.path.join(rst_path, sub_folder)):
                os.mkdir(os.path.join(rst_path, sub_folder))

            rst_file = os.path.join(rst_path, sub_folder, vidname[:7] + '_MDMO_feature.xls')
            workbook = xlwt.Workbook(encoding='utf-8')
            worksheet = workbook.add_sheet('sheet1')

            worksheet.write(0, 0, '#frames')
            worksheet.write(0, 1, 'ROI1_MDMO_rhos')
            worksheet.write(0, 2, 'ROI1_MDMO_angles')
            worksheet.write(0, 3, 'ROI2_MDMO_rhos')
            worksheet.write(0, 4, 'ROI2_MDMO_angles')
            worksheet.write(0, 5, 'ROI3_MDMO_rhos')
            worksheet.write(0, 6, 'ROI3_MDMO_angles')
            worksheet.write(0, 7, 'ROI4_MDMO_rhos')
            worksheet.write(0, 8, 'ROI4_MDMO_angles')
            worksheet.write(0, 9, 'ROI5_MDMO_rhos')
            worksheet.write(0, 10, 'ROI5_MDMO_angles')
            worksheet.write(0, 11, 'ROI6_MDMO_rhos')
            worksheet.write(0, 12, 'ROI6_MDMO_angles')
            worksheet.write(0, 13, 'ROI7_MDMO_rhos')
            worksheet.write(0, 14, 'ROI7_MDMO_angles')
            worksheet.write(0, 15, 'ROI8_MDMO_rhos')
            worksheet.write(0, 16, 'ROI8_MDMO_angles')
            worksheet.write(0, 17, 'ROI9_MDMO_rhos')
            worksheet.write(0, 18, 'ROI9_MDMO_angles')
            worksheet.write(0, 19, 'ROI10_MDMO_rhos')
            worksheet.write(0, 20, 'ROI10_MDMO_angles')
            worksheet.write(0, 21, 'ROI11_MDMO_rhos')
            worksheet.write(0, 22, 'ROI11_MDMO_angles')

            for index in range(1, len(ROI1_MDMO_rhos) + 1):
                worksheet.write(index, 0, index)
                worksheet.write(index, 1, ROI1_MDMO_rhos[index - 1])
                worksheet.write(index, 2, ROI1_MDMO_angles[index - 1])
                worksheet.write(index, 3, ROI2_MDMO_rhos[index - 1])
                worksheet.write(index, 4, ROI2_MDMO_angles[index - 1])
                worksheet.write(index, 5, ROI3_MDMO_rhos[index - 1])
                worksheet.write(index, 6, ROI3_MDMO_angles[index - 1])
                worksheet.write(index, 7, ROI4_MDMO_rhos[index - 1])
                worksheet.write(index, 8, ROI4_MDMO_angles[index - 1])
                worksheet.write(index, 9, ROI5_MDMO_rhos[index - 1])
                worksheet.write(index, 10, ROI5_MDMO_angles[index - 1])
                worksheet.write(index, 11, ROI6_MDMO_rhos[index - 1])
                worksheet.write(index, 12, ROI6_MDMO_angles[index - 1])
                worksheet.write(index, 13, ROI7_MDMO_rhos[index - 1])
                worksheet.write(index, 14, ROI7_MDMO_angles[index - 1])
                worksheet.write(index, 15, ROI8_MDMO_rhos[index - 1])
                worksheet.write(index, 16, ROI8_MDMO_angles[index - 1])
                worksheet.write(index, 17, ROI9_MDMO_rhos[index - 1])
                worksheet.write(index, 18, ROI9_MDMO_angles[index - 1])
                worksheet.write(index, 19, ROI10_MDMO_rhos[index - 1])
                worksheet.write(index, 20, ROI10_MDMO_angles[index - 1])
                worksheet.write(index, 21, ROI11_MDMO_rhos[index - 1])
                worksheet.write(index, 22, ROI11_MDMO_angles[index - 1])

            workbook.save(rst_file)


def main():
    save_MDMO_features_for_CASme2()


if __name__ == '__main__':
    main()
