import dlib
import os
import numpy as np
import math
import xlrd
import xlwt


p68 = "/landmark_model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor_68 = dlib.shape_predictor(p68)

def ChiSquare_dist(np1, np2):
    np_shape = np1.shape[0]
    dist = 0.0
    for i in range(0, np_shape):
        if (np1[i] + np2[i] == 0):
            xs = 0.0
        else:
            xs = (np1[i] - np2[i])*(np1[i] - np2[i]) / (np1[i] + np2[i])

        dist = dist + xs

    return dist


def calculate_distance_block(curr_fea_blks, hf_fea_blks, tf_fea_blks):

    dist_list = []

    for iblock in range(0, 36):
        lbp_feat_1 = curr_fea_blks[iblock]
        lbp_feat_2 = hf_fea_blks[iblock]
        lbp_feat_3 = tf_fea_blks[iblock]

        avg_feat = np.add(lbp_feat_2, lbp_feat_3) / 2
        dist = ChiSquare_dist(lbp_feat_1, avg_feat)

        if (math.isnan(dist)):
            dist = 0
        dist_list.append(dist)

    dist_list.sort()

    sum_dist = 0.0
    for i in range(24, 36):
        sum_dist = sum_dist + dist_list[i]
    return sum_dist


def calculate_contrast_difference(dataset, video_file, window_len):
    print("Starting !!")
    video_name = video_file.split('/')[-1]
    if dataset == 'CASme2':
        feature_path = '/results/features/CAS(ME)^2_LBP_features' + '/' + 's' + video_name[:2] + '/' + video_name[:-4] + '_features.xls'
    elif dataset == 'SAMM Long Videos':
        feature_path = '/results/features/SAMM_Long_Videos_LBP_features' + '/' + video_name + '_features.xls'
    else:
        print("Error! The parameter 'dataset' has two options: 'CASme2' and 'SAMM Long Videos'.")
    
    data = xlrd.open_workbook(feature_path)
    
    lbp_feat_blocks = []
    for i_sheet in range(0, 37):
        feature_data = data.sheet_by_name('sheet' + str(i_sheet))
        num_frame = feature_data.nrows - 1

        feature = []
        for i_row in range(1, num_frame+1):
            feature_frame = np.zeros(256)
            for i_col in range(1, 257):
                feature_frame[i_col-1] = feature_data.cell_value(i_row, i_col)
            feature.append(feature_frame)

        if i_sheet == 0:
            lbp_feat_img = feature
        if (i_sheet >= 1) and (i_sheet <= 36):
            lbp_feat_blocks.append(feature)
        
    features = [lbp_feat_img, lbp_feat_blocks]

    features_frame_block = []
    for i_frame in range(0, num_frame):
        features_frame = []
        for i_blk in range(0, 36):
            features_frame.append(features[1][i_blk][i_frame])
        features_frame_block.append(features_frame)

    print('Number of frames in this video: ', num_frame)
    Farr = np.zeros(num_frame)
    Carr = np.zeros(num_frame)
    k = int(window_len / 2)

    for i in range(k, num_frame - k):
        curr_fea_blks = features_frame_block[i]
        tf_fea_blks = features_frame_block[i-k]
        hf_fea_blks = features_frame_block[i+k]
        dist = calculate_distance_block(curr_fea_blks, hf_fea_blks, tf_fea_blks)
        Farr[i] = dist
    print("F array has been calculated completely.")
    for i in range(k, num_frame - k):
        Carr[i] = Farr[i] - 0.5 * (Farr[i - k] + Farr[i + k])
        if (Carr[i] < 0):
            Carr[i] = 0

    print('Farr = ', Farr)
    print('Carr = ', Carr)

    return Farr, Carr


def save_contrast_differences_for_CASme2(window_len):
    rst_path = '/results/contrast_differences/CAS(ME)^2_LBP_contrast_differences' + '/' + 'window_len_' + str(window_len)
    if not os.path.exists(rst_path):
        os.mkdir(rst_path)
    folder_data = " "   # path of CAS(ME)^2 dataset
    subfolders = os.listdir(folder_data)

    for sub_folder in subfolders:
        vidfolders = os.listdir(os.path.join(folder_data, sub_folder))
        for vidname in vidfolders:
            print(vidname)
            Farr, Carr = calculate_contrast_difference('CASme2', os.path.join(folder_data, sub_folder, vidname), window_len)

            # save results
            if not os.path.exists(os.path.join(rst_path, sub_folder)):
                os.mkdir(os.path.join(rst_path, sub_folder))
            if not os.path.exists(os.path.join(rst_path, sub_folder, vidname[:-4])):
                os.mkdir(os.path.join(rst_path, sub_folder, vidname[:-4]))

            rst_file = os.path.join(rst_path, sub_folder, vidname[:-4], 'contrast_differences.xls')
            workbook = xlwt.Workbook(encoding='utf-8')
            worksheet = workbook.add_sheet('sheet1')

            worksheet.write(0, 0, '#frames')
            worksheet.write(0, 1, 'Farr')
            worksheet.write(0, 2, 'Carr')

            for index in range(1, len(Carr)+1):
                worksheet.write(index, 0, index)
                worksheet.write(index, 1, Farr[index - 1])
                worksheet.write(index, 2, Carr[index - 1])
            workbook.save(rst_file)


def save_contrast_differences_for_SAMM_Long_Videos(window_len):
    rst_path = '/results/contrast_differences/SAMM_Long_Videos_LBP_contrast_differences' + '/' + 'window_len_' + str(window_len)
    if not os.path.exists(rst_path):
        os.mkdir(rst_path)
    folder_data = " "  # path of SAMM Long Videos dataset
    vidfolders = os.listdir(folder_data)

    for vidname in vidfolders:
        print(vidname)
        Farr, Carr = calculate_contrast_difference('SAMM Long Videos', os.path.join(folder_data, vidname), window_len)

        if not os.path.exists(os.path.join(rst_path, vidname)):
            os.mkdir(os.path.join(rst_path, vidname))
        rst_file = os.path.join(rst_path, vidname, 'contrast_differences.xls')
        workbook = xlwt.Workbook(encoding='utf-8')
        worksheet = workbook.add_sheet('sheet1')

        worksheet.write(0, 0, '#frames')
        worksheet.write(0, 1, 'Farr')
        worksheet.write(0, 2, 'Carr')

        for index in range(1, len(Carr)+1):
            worksheet.write(index, 0, index)
            worksheet.write(index, 1, Farr[index - 1])
            worksheet.write(index, 2, Carr[index - 1])
        workbook.save(rst_file)


def main():
    # save_contrast_differences_for_CASme2(window_len=11)
    # save_contrast_differences_for_CASme2(window_len=15)
    save_contrast_differences_for_SAMM_Long_Videos(window_len=57)
    # save_contrast_differences_for_SAMM_Long_Videos(window_len=89)


if __name__ == '__main__':
    main()
