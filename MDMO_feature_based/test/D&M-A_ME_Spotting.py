'''
The code here refers to the code provided by the MEGC2020 competition paper
"Spatio-temporal fusion for Macro- and Micro-expression Spotting in Long Video Sequences"
which can be downloaded in "https://github.com/Small-Steel-Cannon/ MEGC2020",
and some parts have been slightly modified to fit our model.
'''

import os
import numpy as np
import datetime
from scipy import signal
import pandas as pd
import xlrd


def wave_detect(y1, y2, y3, y4, y5, y6, first_interval, angle_condition):
    ys = [y1, y2, y3, y4, y5, y6]
    ys_max = []
    for i_y in range(len(ys)):
        ys_max.append(np.max(ys[i_y]))

    selected_iy = []
    for i_y in range(len(ys_max)):
        if ys_max[i_y] >= 1.1:
            selected_iy.append(i_y)

    if len(selected_iy) == 0:
        return None

    selected_ys = []
    selected_ys_max = []
    diffs = []
    for idx in range(len(selected_iy)):
        selected_ys.append(ys[selected_iy[idx]])
        selected_ys_max.append(ys_max[selected_iy[idx]])
        diffs.append(ys[selected_iy[idx]] - np.ones(len(ys[selected_iy[idx]])) * 0.1 * ys_max[selected_iy[idx]])

    detect_list = []
    for i in range(0, len(diffs[0])):
        abs_diffs = []
        diffs_sign = []
        for idx in range(len(diffs)):
            abs_diffs.append(abs(diffs[idx][i]))
            diffs_sign.append(diffs[idx][i]/abs(diffs[idx][i]))

        diffs_pos_idx = [idx for idx in range(0, len(diffs_sign)) if diffs_sign[idx] == 1]
        diffs_neg_idx = [idx for idx in range(0, len(diffs_sign)) if diffs_sign[idx] == -1]
        if len(diffs_pos_idx) > len(diffs_neg_idx):
            center = i
            lefts = []
            rights = []
            for idx in range(len(diffs_pos_idx)):
                if center + 1 <= selected_ys[diffs_pos_idx[idx]].shape[0] - 1 and center - 1 >= 0:
                    if selected_ys[diffs_pos_idx[idx]][center] > 0.3 and selected_ys[diffs_pos_idx[idx]][center + 1] - selected_ys[diffs_pos_idx[idx]][center] < 0 and selected_ys[diffs_pos_idx[idx]][center - 1] - selected_ys[diffs_pos_idx[idx]][center] < 0:
                        # ==========left==========
                        left = center
                        while selected_ys[diffs_pos_idx[idx]][left - 1] - selected_ys[diffs_pos_idx[idx]][left] < 0:
                            left -= 1
                            if left - 1 < 0:
                                break
                        # ==========right=========
                        right = center
                        while selected_ys[diffs_pos_idx[idx]][right + 1] - selected_ys[diffs_pos_idx[idx]][right] < 0:
                            right += 1
                            if right + 1 > selected_ys[diffs_pos_idx[idx]].shape[0] - 1:
                                break
                        lefts.append(left)
                        rights.append(right)

            if len(lefts) > 0:
                left_fusion = int(round(np.mean(lefts)))
                right_fusion = int(round(np.mean(rights)))
                length = right_fusion - left_fusion + 1

                if length <= 8:
                    pos_angle_frame_num = 0
                    neg_angle_frame_num = 0
                    for idx in range(0, length):
                        if left_fusion + first_interval + idx <= angle_condition.shape[0] - 1:
                            if angle_condition[left_fusion + first_interval + idx] > 0:
                                pos_angle_frame_num += 1
                            elif angle_condition[left_fusion + first_interval + idx] < 0:
                                neg_angle_frame_num += 1
                    angle_diff = abs(pos_angle_frame_num - neg_angle_frame_num)
                    if angle_diff <= 0:
                        detect_list.append([left_fusion + first_interval + 1, right_fusion + first_interval + 1])

        elif len(diffs_pos_idx) == len(diffs_neg_idx):
            abs_pos_sum = 0
            abs_neg_sum = 0
            for idx in diffs_pos_idx:
                abs_pos_sum = abs_pos_sum + abs_diffs[idx]
            for idx in diffs_neg_idx:
                abs_neg_sum = abs_neg_sum + abs_diffs[idx]
            if abs_pos_sum >= abs_neg_sum:
                center = i
                lefts = []
                rights = []
                for idx in range(len(diffs_pos_idx)):
                    if center + 1 <= selected_ys[diffs_pos_idx[idx]].shape[0] - 1 and center - 1 >= 0:
                        if selected_ys[diffs_pos_idx[idx]][center] > 0.3 and selected_ys[diffs_pos_idx[idx]][center + 1] - selected_ys[diffs_pos_idx[idx]][center] < 0 and selected_ys[diffs_pos_idx[idx]][center - 1] - selected_ys[diffs_pos_idx[idx]][center] < 0:
                            # ==========left==========
                            left = center
                            while selected_ys[diffs_pos_idx[idx]][left - 1] - selected_ys[diffs_pos_idx[idx]][left] < 0:
                                left -= 1
                                if left - 1 < 0:
                                    break
                            # ==========right=========
                            right = center
                            while selected_ys[diffs_pos_idx[idx]][right + 1] - selected_ys[diffs_pos_idx[idx]][right] < 0:
                                right += 1
                                if right + 1 > selected_ys[diffs_pos_idx[idx]].shape[0] - 1:
                                    break
                            lefts.append(left)
                            rights.append(right)

                if len(lefts) > 0:
                    left_fusion = int(round(np.mean(lefts)))
                    right_fusion = int(round(np.mean(rights)))
                    length = right_fusion - left_fusion + 1

                    if length <= 8:
                        pos_angle_frame_num = 0
                        neg_angle_frame_num = 0
                        for idx in range(0, length):
                            if left_fusion + first_interval + idx <= angle_condition.shape[0] - 1:
                                if angle_condition[left_fusion + first_interval + idx] > 0:
                                    pos_angle_frame_num += 1
                                elif angle_condition[left_fusion + first_interval + idx] < 0:
                                    neg_angle_frame_num += 1
                        angle_diff = abs(pos_angle_frame_num - neg_angle_frame_num)
                        if angle_diff <= 0:
                            detect_list.append([left_fusion + first_interval + 1, right_fusion + first_interval + 1])

    detect_list = np.array(detect_list)
    return detect_list


def detect(feature, parameter, window_len_1, window_len_2, kl_2, kr_2, window_len_3, kl_3, kr_3, window_len_4, window_len_5, kl_5, kr_5, window_len_6, kl_6, kr_6):
    first_interval = 100
    feature = feature[:, 1:]
    col_count = feature.shape[1]
    Predict_list = []

    for i in range(col_count):
        if i % 2 == 0:
            current_line = feature[:, i:i + 1]
            current_line = current_line[:]
            relative_difference_1 = np.zeros(len(current_line))
            relative_difference_2 = np.zeros(len(current_line))
            relative_difference_3 = np.zeros(len(current_line))
            relative_difference_4 = np.zeros(len(current_line))
            relative_difference_5 = np.zeros(len(current_line))
            relative_difference_6 = np.zeros(len(current_line))

            k_1 = int(window_len_1/2)
            for idx in range(k_1, len(current_line)-k_1):
                relative_difference_1[idx] = current_line[idx] - 0.5*(current_line[idx-k_1] + current_line[idx+k_1])
            relative_difference_1 = relative_difference_1[first_interval:-first_interval]
            relative_difference_1 = relative_difference_1.reshape(-1)

            for idx in range(kl_2, len(current_line)-kr_2):
                relative_difference_2[idx] = current_line[idx] - 0.5*(current_line[idx-kl_2] + current_line[idx+kr_2])
            relative_difference_2 = relative_difference_2[first_interval:-first_interval]
            relative_difference_2 = relative_difference_2.reshape(-1)

            for idx in range(kl_3, len(current_line)-kr_3):
                relative_difference_3[idx] = current_line[idx] - 0.5*(current_line[idx-kl_3] + current_line[idx+kr_3])
            relative_difference_3 = relative_difference_3[first_interval:-first_interval]
            relative_difference_3 = relative_difference_3.reshape(-1)

            k_4 = int(window_len_4 / 2)
            for idx in range(k_4, len(current_line) - k_4):
                relative_difference_4[idx] = current_line[idx] - 0.5 * (current_line[idx - k_4] + current_line[idx + k_4])
            relative_difference_4 = relative_difference_4[first_interval:-first_interval]
            relative_difference_4 = relative_difference_4.reshape(-1)

            for idx in range(kl_5, len(current_line)-kr_5):
                relative_difference_5[idx] = current_line[idx] - 0.5*(current_line[idx-kl_5] + current_line[idx+kr_5])
            relative_difference_5 = relative_difference_5[first_interval:-first_interval]
            relative_difference_5 = relative_difference_5.reshape(-1)

            for idx in range(kl_6, len(current_line)-kr_6):
                relative_difference_6[idx] = current_line[idx] - 0.5*(current_line[idx-kl_6] + current_line[idx+kr_6])
            relative_difference_6 = relative_difference_6[first_interval:-first_interval]
            relative_difference_6 = relative_difference_6.reshape(-1)

            y_hat_1 = signal.savgol_filter(relative_difference_1, parameter, 3)
            y_hat_2 = signal.savgol_filter(relative_difference_2, parameter, 3)
            y_hat_3 = signal.savgol_filter(relative_difference_3, parameter, 3)
            y_hat_4 = signal.savgol_filter(relative_difference_4, parameter, 3)
            y_hat_5 = signal.savgol_filter(relative_difference_5, parameter, 3)
            y_hat_6 = signal.savgol_filter(relative_difference_6, parameter, 3)

            angle_condition = feature[:, i+1:i+2]
            detect_list = wave_detect(y_hat_1, y_hat_2, y_hat_3, y_hat_4, y_hat_5, y_hat_6, first_interval, angle_condition)
            if not detect_list is None:
                Predict_list.append(detect_list)
    return Predict_list


def get_spot_list(res):
    first = False
    start = None
    end = None
    spot_list = []
    for i in range(res.shape[0]):
        if res[i] == 1 and first == False:
            first = True
            start = i
        if res[i] == 0 and first == True:
            end = i - 1
            spot_list.append([start + 1, end + 1])
            first = False
        if i == res.shape[0] - 1 and first == True:
            end = i
            spot_list.append([start + 1, end + 1])
    return spot_list


def generate_label(Predict_list, frame_count):
    res = np.zeros([frame_count, ], dtype=np.int8)
    for i in range(len(Predict_list)):
        current_predict = Predict_list[i]
        if current_predict.shape[0] > 30:
            continue
        for k in range(current_predict.shape[0]):
            son_predict = current_predict[k]
            left = int(son_predict[0])
            right = int(son_predict[1])
            length = right - left + 1
            for j in range(length):
                res[left + j - 1] = 1

    spot_list = get_spot_list(res)

    new_spot_list = []
    idx = 0
    while idx <= len(spot_list) - 1:
        continuous_count = 0
        while idx + continuous_count + 1 <= len(spot_list) - 1:
            if spot_list[idx + continuous_count + 1][0] - spot_list[idx + continuous_count][1] <= 2:
                continuous_count += 1
            else:
                break
        new_spot = [spot_list[idx][0], spot_list[idx + continuous_count][1]]  # 融合的操作
        new_spot_list.append(new_spot)
        idx = idx + continuous_count + 1

    return new_spot_list


def get_page(label_path):
    excel = pd.ExcelFile(label_path)
    pages = excel.sheet_names
    page0 = np.array(pd.read_excel(label_path, pages[0], header=None))
    page1 = np.array(pd.read_excel(label_path, pages[1], header=None))
    page2 = np.array(pd.read_excel(label_path, pages[2], header=None))
    return page0, page1, page2


def read_label(s, code, page0, page1, page2):
    column0 = page1[:, 0:1].reshape((-1))
    s = int(s)
    idx = np.argwhere(column0 == s)
    # print(idx)
    first_idx = idx[0][0]
    convert_s = page1[first_idx][2]

    column0 = page2[:, 0:1].reshape((-1))
    idx = np.argwhere(column0 == int(code[1:]))
    first_idx = idx[0][0]
    convert_code = page2[first_idx][1]

    column0 = page0[:, 0:1].reshape((-1))
    idx = np.argwhere(column0 == convert_s)
    column1 = page0[:, 1:2].reshape((-1))
    column1 = column1[idx[0][0]: idx[-1][0] + 1]
    base_idx = idx[0][0]
    for i in range(column1.shape[0]):
        column1[i] = column1[i][0:-2]
        if column1[i][-1] == '_':
            column1[i] = column1[i][0:-1]
    idx = np.argwhere(column1 == convert_code)
    express_list = []
    for i in range(idx.shape[0]):
        current = idx[i][0] + base_idx
        if page0[current][7] == 'micro-expression':
            start = page0[current][2]
            peak = page0[current][3]
            end = page0[current][4]
            if end == 0:
                end = peak
            if start == -1 or peak == -1 or end == -1:
                continue
            express_list.append([start, end])

    return express_list


def analyse(label_list, spot_list):
    FN_count = 0
    TP_count = 0
    FP_count = 0
    TP_lsit = [0] * len(spot_list)
    FN_list = [1] * len(label_list)
    class_list = []
    main_item = None

    for i in range(len(spot_list)):
        tmp_spot = spot_list[i]
        for j in range(len(label_list)):
            tmp_label = label_list[j]
            if tmp_spot[1] >= tmp_label[1]:
                right_range = tmp_spot
                left_range = tmp_label
            else:
                right_range = tmp_label
                left_range = tmp_spot
            flag = left_range[1] - right_range[0]

            if flag <= 0:
                continue

            if flag > 0:
                if left_range[0] >= right_range[0]:
                    inter_length = left_range[1] - left_range[0] + 1
                    union_length = right_range[1] - right_range[0] + 1
                if left_range[0] < right_range[0]:
                    inter_length = left_range[1] - right_range[0] + 1
                    union_length = right_range[1] - left_range[0] + 1

                K = inter_length / union_length  # IOU

                if K >= 0.5:
                    TP_lsit[i] = 1
                    TP_count += 1
                    FN_list[j] = 0
                    tmp_label = label_list[j].copy()
                    tmp_label.append(tmp_spot[0])
                    tmp_label.append(tmp_spot[1])
                    tmp_label.append(K)
                    tmp_item = np.array([tmp_label])

                    if main_item is None:
                        main_item = tmp_item
                        class_list.append('TP')
                    else:
                        main_item = np.concatenate([main_item, tmp_item])
                        class_list.append('TP')

    for i in range(len(label_list)):
        if FN_list[i] == 1:
            FN_count += 1
            tmp = label_list[i].copy()
            tmp.append(-1)
            tmp.append(-1)
            tmp.append(0)
            tmp = np.array([tmp])
            if main_item is None:
                main_item = tmp
                class_list.append('FN')
            else:
                main_item = np.concatenate([main_item, tmp])
                class_list.append('FN')

    for i in range(len(spot_list)):
        if TP_lsit[i] == 0:
            FP_count += 1
            tmp = spot_list[i].copy()
            tmp.insert(0, -1)
            tmp.insert(1, -1)
            tmp.append(0)
            tmp = np.array([tmp])
            if main_item is None:
                main_item = tmp
                class_list.append('FP')
            else:
                main_item = np.concatenate([main_item, tmp])
                class_list.append('FP')

    return TP_count, FP_count, FN_count, main_item, class_list


def spotting(label_path, base_path, parameter_path, window_len_1, window_len_2, kl_2, kr_2, window_len_3, kl_3, kr_3, window_len_4, window_len_5, kl_5, kr_5, window_len_6, kl_6, kr_6):
    TP = 0
    FP = 0
    FN = 0
    total_P = 0
    DF = None
    parameter_count = -1
    no_spot = 0
    no_spot_list = []

    # read parameters from excel
    parameter_list = pd.read_excel(parameter_path)
    parameter_list = np.array(parameter_list)
    parameter_list = parameter_list[:, -1]
    parameter_list = parameter_list.reshape(-1)
    print(parameter_list)

    page0, page1, page2 = get_page(label_path)
    print(datetime.datetime.now())

    dirs = os.listdir(base_path)
    for i in range(len(dirs)):
        dir_path = base_path + '\\' + dirs[i]
        files = os.listdir(dir_path)
        for j in range(len(files)):
            parameter_count += 1
            file_path = dir_path + '\\' + files[j]
            print(file_path)
            no_spot += 1
            no_spot_list.append(files[j][:7])

            s = files[j][0:2]
            code = files[j][3:7]

            p0 = page0.copy()
            p1 = page1.copy()
            p2 = page2.copy()
            current_label = read_label(s, code, p0, p1, p2)
            total_P += len(current_label)

            # read feature
            current_feature = pd.read_excel(file_path)
            current_feature = np.array(current_feature)

            Predict_list = detect(current_feature, parameter_list[parameter_count], window_len_1, window_len_2, kl_2, kr_2, window_len_3, kl_3, kr_3,
                                  window_len_4, window_len_5, kl_5, kr_5, window_len_6, kl_6, kr_6)
            spot_list = generate_label(Predict_list, current_feature.shape[0])

            if len(spot_list) == 0:
                current_TP = 0
                current_FP = 0
                current_FN = len(current_label)
                current_main_item = None
                current_class_list = []
                for idx in range(len(current_label)):
                    tmp = current_label[idx].copy()
                    tmp.append(-1)
                    tmp.append(-1)
                    tmp.append(0)
                    tmp = np.array([tmp])
                    if current_main_item is None:
                        current_main_item = tmp
                        current_class_list.append('FN')
                    else:
                        current_main_item = np.concatenate([current_main_item, tmp])
                        current_class_list.append('FN')
            else:
                current_TP, current_FP, current_FN, current_main_item, current_class_list = analyse(current_label, spot_list)

            # record spotting results
            if not current_main_item is None:
                TP += current_TP
                FP += current_FP
                FN += current_FN
                print(current_TP, current_FP, current_FN)

                current_index = [files[j][:7]] * current_main_item.shape[0]
                df = pd.DataFrame(current_main_item, index=current_index, columns=['T_1', 'T_2', 'P_1', 'P_2', 'K'])
                df.insert(len(df.columns), 'class', current_class_list)
                df["T_1"] = df["T_1"].astype("int")
                df["T_2"] = df["T_2"].astype("int")
                df["P_1"] = df["P_1"].astype("int")
                df["P_2"] = df["P_2"].astype("int")
                if DF is None:
                    DF = df
                else:
                    DF = pd.concat([DF, df], axis=0)

    P = total_P
    print('total_P', total_P)
    if P - TP == FN:
        print('P - TP == FN')
    print('TP:', TP)
    print('FP:', FP)
    print('FN:', FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * recall * precision / (recall + precision)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1: ', f1)

    print('no_spot', no_spot)
    print('no_spot_list', no_spot_list)

    # save spotting results
    rst_save_path = os.getcwd() + r'\spotting_results_record.xlsx'
    DF.to_excel(rst_save_path)


def main():
    label_path = r'D:\ME_Database_Download\CAS(ME)^2\CAS(ME)^2code_final.xlsx'  # path of CAS(ME)^2 label file
    MDMO_features_path = r'..\..\..\results\features\CAS(ME)^2_11ROIs_MDMO_features'
    parameter_path = os.getcwd() + r'\Savitzky-golay_parameters.xlsx'
    spotting(label_path, MDMO_features_path, parameter_path, window_len_1=11, window_len_2=11, kl_2=4, kr_2=6, window_len_3=11, kl_3=6, kr_3=4, window_len_4=15, window_len_5=15, kl_5=6, kr_5=8, window_len_6=15, kl_6=8, kr_6=6)


if __name__ == '__main__':
    main()
