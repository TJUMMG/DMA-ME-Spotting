import os
import xlrd
import numpy as np
import math
import pandas as pd


def calculate_IOU(interval1, interval2):
    if interval1[1] < interval2[0]:
        IOU = 0
    elif (interval1[1] >= interval2[0]) and (interval1[1] <= interval2[1]):
        if interval1[0] < interval2[0]:
            IOU = (interval1[1] - interval2[0] + 1) / (interval2[1] - interval1[0] + 1)
        else:
            IOU = (interval1[1] - interval1[0] + 1) / (interval2[1] - interval2[0] + 1)
    elif (interval1[1] > interval2[1]) and (interval1[0] <= interval2[1]):
        if interval1[0] >= interval2[0]:
            IOU = (interval2[1] - interval1[0] + 1) / (interval1[1] - interval2[0] + 1)
        else:
            IOU = (interval2[1] - interval2[0] + 1) / (interval1[1] - interval1[0] + 1)
    else:
        IOU = 0

    return IOU


def get_GT_intervals_for_video(sub, video_name):
    workpath ="datasets/CAS(ME)^2/CAS(ME)^2code_final.xlsx"  # path of CAS(ME)^2 label file
    data = xlrd.open_workbook(workpath)
    code_final = data.sheet_by_name('CASMEcode_final')
    name_rule1 = data.sheet_by_name('naming rule1')
    name_rule2 = data.sheet_by_name('naming rule2')

    rowNum_nr1 = name_rule1.nrows
    for i in range(0, rowNum_nr1):
        if name_rule1.cell_value(i, 0) == sub:
            i_sub = name_rule1.cell_value(i, 2)
            break

    rowNum_nr2 = name_rule2.nrows
    for i in range(0, rowNum_nr2):
        if name_rule2.cell_value(i, 0) == video_name[3:7]:
            i_expression = name_rule2.cell_value(i, 1)
            break

    interval_list = []
    rowNum_cf = code_final.nrows
    for i in range(0, rowNum_cf):
        if (code_final.cell_value(i, 0) == i_sub) and (code_final.cell_value(i, 1)[:-2] == i_expression or code_final.cell_value(i, 1)[:-3] == i_expression) and (code_final.cell_value(i, 7) == 'micro-expression'):
            onset = code_final.cell_value(i, 2)
            apex = code_final.cell_value(i, 3)
            offset = code_final.cell_value(i, 4)

            if offset == 0:
                interval = [onset, apex + apex - onset]
            else:
                interval = [onset, offset]

            interval_list.append(interval)

    return interval_list


def get_spotted_interval_for_video(video_name, window_len_1, kl_1, kr_1, peakl_1, peakr_1, p_1, window_len_2, kl_2, kr_2, peakl_2, peakr_2, p_2, window_len_3, peaklr_len_3, p_3, window_len_4, kl_4, kr_4, peakl_4, peakr_4, p_4, window_len_5, kl_5, kr_5, peakl_5, peakr_5, p_5, window_len_6, peaklr_len_6, p_6):
    # scale1-asymmetric1
    rst_path_1 ='/results/contrast_differences/CAS(ME)^2_LBP_contrast_differences' + '/' + 'window_len_' + str(window_len_1) + '_kl_' + str(kl_1) + '_kr_' + str(kr_1)
    workpath_1 = rst_path_1 + '/' + 's' + video_name[:2] + '/' + video_name[:-4] + '/' + 'contrast_differences.xls'
    data_1 = xlrd.open_workbook(workpath_1)
    data_v_1 = data_1.sheet_by_name('sheet1')
    rowNum_dv_1 = data_v_1.nrows
    Carr_1 = []
    for i in range(1, rowNum_dv_1):
        Carr_1.append(data_v_1.cell_value(i, 2))

    Cmean_1 = np.sum(Carr_1[2*kl_1:len(Carr_1)-kr_1*2])/len(Carr_1)
    Cmax_1 = np.max(Carr_1[2*kl_1:len(Carr_1)-kr_1*2])
    epsilon_1 = 0.01 * p_1
    # threshold
    Thr_1 = Cmean_1 + epsilon_1 * (Cmax_1 - Cmean_1)

    # scale1-asymmetric2
    rst_path_2 ='/results/contrast_differences/CAS(ME)^2_LBP_contrast_differences' + '/' + 'window_len_' + str(window_len_2) + '_kl_' + str(kl_2) + '_kr_' + str(kr_2)
    workpath_2 = rst_path_2 + '/' + 's' + video_name[:2] + '/' + video_name[:-4] + '/' + 'contrast_differences.xls'
    data_2 = xlrd.open_workbook(workpath_2)
    data_v_2 = data_2.sheet_by_name('sheet1')
    rowNum_dv_2 = data_v_2.nrows
    Carr_2 = []
    for i in range(1, rowNum_dv_2):
        Carr_2.append(data_v_2.cell_value(i, 2))

    Cmean_2 = np.sum(Carr_2[2*kl_2:len(Carr_2)-kr_2*2])/len(Carr_2)
    Cmax_2 = np.max(Carr_2[2*kl_2:len(Carr_2)-kr_2*2])
    epsilon_2 = 0.01 * p_2
    # threshold
    Thr_2 = Cmean_2 + epsilon_2 * (Cmax_2 - Cmean_2)

    # scale1-symmetric
    rst_path_3 = '/results/contrast_differences/CAS(ME)^2_LBP_contrast_differences' + '/' + 'window_len_' + str(window_len_3)
    workpath_3 = rst_path_3 + '/' + 's' + video_name[:2] + '/' + video_name[:-4] + '/' + 'contrast_differences.xls'
    data_3 = xlrd.open_workbook(workpath_3)
    data_v_3 = data_3.sheet_by_name('sheet1')
    rowNum_dv_3 = data_v_3.nrows
    Carr_3 = []
    for i in range(1, rowNum_dv_3):
        Carr_3.append(data_v_3.cell_value(i, 2))

    Cmean_3 = np.sum(Carr_3[int(window_len_3/2)*2:len(Carr_3)-int(window_len_3/2)*2])/len(Carr_3)
    Cmax_3 = np.max(Carr_3[int(window_len_3/2)*2:len(Carr_3)-int(window_len_3/2)*2])
    epsilon_3 = 0.01 * p_3
    # threshold
    Thr_3 = Cmean_3 + epsilon_3 * (Cmax_3 - Cmean_3)

    # scale2-asymmetric1
    rst_path_4 = '/results/contrast_differences/CAS(ME)^2_LBP_contrast_differences' + '/' + 'window_len_' + str(window_len_4) + '_kl_' + str(kl_4) + '_kr_' + str(kr_4)
    workpath_4 = rst_path_4 + '/' + 's' + video_name[:2] + '/' + video_name[:-4] + '/' + 'contrast_differences.xls'
    data_4 = xlrd.open_workbook(workpath_4)
    data_v_4 = data_4.sheet_by_name('sheet1')
    rowNum_dv_4 = data_v_4.nrows
    Carr_4 = []
    for i in range(1, rowNum_dv_4):
        Carr_4.append(data_v_4.cell_value(i, 2))

    Cmean_4 = np.sum(Carr_4[2*kl_4:len(Carr_4)-kr_4*2])/len(Carr_4)
    Cmax_4 = np.max(Carr_4[2*kl_4:len(Carr_4)-kr_4*2])
    epsilon_4 = 0.01 * p_4
    # threshold
    Thr_4 = Cmean_4 + epsilon_4 * (Cmax_4 - Cmean_4)

    # scale2-asymmetric2
    rst_path_5 ='/results/contrast_differences/CAS(ME)^2_LBP_contrast_differences' + '/' + 'window_len_' + str(window_len_5) + '_kl_' + str(kl_5) + '_kr_' + str(kr_5)
    workpath_5 = rst_path_5 + '/' + 's' + video_name[:2] + '/' + video_name[:-4] + '/' + 'contrast_differences.xls'
    data_5 = xlrd.open_workbook(workpath_5)
    data_v_5 = data_5.sheet_by_name('sheet1')
    rowNum_dv_5 = data_v_5.nrows
    Carr_5 = []
    for i in range(1, rowNum_dv_5):
        Carr_5.append(data_v_5.cell_value(i, 2))

    Cmean_5 = np.sum(Carr_5[2*kl_5:len(Carr_5)-kr_5*2])/len(Carr_5)
    Cmax_5 = np.max(Carr_5[2*kl_5:len(Carr_5)-kr_5*2])
    epsilon_5 = 0.01 * p_5
    # threshold
    Thr_5 = Cmean_5 + epsilon_5 * (Cmax_5 - Cmean_5)

    # scale2-symmetric
    rst_path_6 = '/results/contrast_differences/CAS(ME)^2_LBP_contrast_differences' + '/' + 'window_len_' + str(window_len_6)
    workpath_6 = rst_path_6 + '/' + 's' + video_name[:2] + '/' + video_name[:-4] + '/' + 'contrast_differences.xls'
    data_6 = xlrd.open_workbook(workpath_6)
    data_v_6 = data_6.sheet_by_name('sheet1')
    rowNum_dv_6 = data_v_6.nrows
    Carr_6 = []
    for i in range(1, rowNum_dv_6):
        Carr_6.append(data_v_6.cell_value(i, 2))

    Cmean_6 = np.sum(Carr_6[int(window_len_6/2)*2:len(Carr_6)-int(window_len_6/2)*2])/len(Carr_6)
    Cmax_6 = np.max(Carr_6[int(window_len_6/2)*2:len(Carr_6)-int(window_len_6/2)*2])
    epsilon_6 = 0.01 * p_6
    # threshold
    Thr_6 = Cmean_6 + epsilon_6 * (Cmax_6 - Cmean_6)


    # indicator vectors
    diff_1 = Carr_1 - np.ones(len(Carr_1)) * Thr_1
    diff_2 = Carr_2 - np.ones(len(Carr_2)) * Thr_2
    diff_3 = Carr_3 - np.ones(len(Carr_3)) * Thr_3
    diff_4 = Carr_4 - np.ones(len(Carr_4)) * Thr_4
    diff_5 = Carr_5 - np.ones(len(Carr_5)) * Thr_5
    diff_6 = Carr_6 - np.ones(len(Carr_6)) * Thr_6

    peakls = [peakl_1, peakl_2, peaklr_len_3, peakl_4, peakl_5, peaklr_len_6]
    peakrs = [peakr_1, peakr_2, peaklr_len_3, peakr_4, peakr_5, peaklr_len_6]

    res = np.zeros(len(diff_1))
    for i in range(2*np.max([kl_1, kl_2, int(window_len_3/2), kl_4, kl_5, int(window_len_6/2)]), len(diff_1)-2*np.max([kr_1, kr_2, int(window_len_3/2), kr_4, kr_5, int(window_len_6/2)])):
        abs_diffs = [abs(diff_1[i]), abs(diff_2[i]), abs(diff_3[i]), abs(diff_4[i]), abs(diff_5[i]), abs(diff_6[i])]
        diffs_sign = [diff_1[i] / abs(diff_1[i]), diff_2[i] / abs(diff_2[i]), diff_3[i] / abs(diff_3[i]),
                      diff_4[i] / abs(diff_4[i]), diff_5[i] / abs(diff_5[i]), diff_6[i] / abs(diff_6[i])]
        diffs_pos_idx = [i for i in range(0, len(diffs_sign)) if diffs_sign[i] == 1]
        diffs_neg_idx = [i for i in range(0, len(diffs_sign)) if diffs_sign[i] == -1]
        if len(diffs_pos_idx) > len(diffs_neg_idx):
            peakl = 0
            peakr = 0
            for idx in diffs_pos_idx:
                peakl = peakl + peakls[idx]
                peakr = peakr + peakrs[idx]
            peakl = math.floor(peakl / len(diffs_pos_idx))
            peakr = math.floor(peakr / len(diffs_pos_idx))

            res[i] = 1
            j = i + 1
            k = i - 1
            countl = 0
            countr = 0
            while (countl <= peakl - 1):
                countl = countl + 1
                if k >= 0:
                    res[k] = 1
                k = k - 1
            while (countr <= peakr - 1):
                countr = countr + 1
                if j <= len(diff_1) - 1:
                    res[j] = 1
                j = j + 1
        elif len(diffs_pos_idx) == len(diffs_neg_idx):
            abs_pos_sum = 0
            abs_neg_sum = 0
            for idx in diffs_pos_idx:
                abs_pos_sum = abs_pos_sum + abs_diffs[idx]
            for idx in diffs_neg_idx:
                abs_neg_sum = abs_neg_sum + abs_diffs[idx]

            if abs_pos_sum >= abs_neg_sum:
                peakl = 0
                peakr = 0
                for idx in diffs_pos_idx:
                    peakl = peakl + peakls[idx]
                    peakr = peakr + peakrs[idx]
                peakl = math.floor(peakl / len(diffs_pos_idx))
                peakr = math.floor(peakr / len(diffs_pos_idx))

                res[i] = 1
                j = i + 1
                k = i - 1
                countl = 0
                countr = 0
                while (countl <= peakl - 1):
                    countl = countl + 1
                    if k >= 0:
                        res[k] = 1
                    k = k - 1
                while (countr <= peakr - 1):
                    countr = countr + 1
                    if j <= len(diff_1) - 1:
                        res[j] = 1
                    j = j + 1

    print('res = ', res)

    # res to interval
    spot_interval_list = []
    i = 0
    while (i >= 0 and i < len(res)):
        if res[i] == 1:
            interval = []
            interval.append(i)
            while (i < len(res) - 1):
                i = i + 1
                if res[i] == 1:
                    interval.append(i)
                else:
                    break

            if len(interval) <= 34:
                spot_interval_list.append([interval[0], interval[-1]])
        i = i + 1

    return spot_interval_list


def Evaluate_CASme2(window_len_1, kl_1, kr_1, peakl_1, peakr_1, p_1, window_len_2, kl_2, kr_2, peakl_2, peakr_2, p_2, window_len_3, peaklr_len_3, p_3,
                    window_len_4, kl_4, kr_4, peakl_4, peakr_4, p_4, window_len_5, kl_5, kr_5, peakl_5, peakr_5, p_5, window_len_6, peaklr_len_6, p_6):
    folder_data = '/datasets/CAS(ME)^2/rawvideo'   # path of CAS(ME)^2 dataset
    subfolders = os.listdir(folder_data)
    TP = []
    GT_num = []
    SPOT_num = []
    DF = None
    for sub_folder in subfolders:
        vidfolders = os.listdir(os.path.join(folder_data, sub_folder))
        for vidname in vidfolders:
            tp = 0
            print(vidname)
            current_index = [vidname[:7]]

            sub_idx = int(sub_folder[-2:])
            spot_interval_list = get_spotted_interval_for_video(vidname, window_len_1, kl_1, kr_1, peakl_1, peakr_1, p_1, window_len_2, kl_2, kr_2, peakl_2, peakr_2, p_2, window_len_3, peaklr_len_3, p_3,
                                                                window_len_4, kl_4, kr_4, peakl_4, peakr_4, p_4, window_len_5, kl_5, kr_5, peakl_5, peakr_5, p_5, window_len_6, peaklr_len_6, p_6)
            gt_interval_list = get_GT_intervals_for_video(sub_idx, vidname)

            gt_interval_num = len(gt_interval_list)
            spot_interval_num = len(spot_interval_list)
            print(gt_interval_num)
            print(spot_interval_num)
            GT_num.append(gt_interval_num)
            SPOT_num.append(spot_interval_num)

            tp_index = []
            for i in range(0, gt_interval_num):
                IOU_list = []
                for j in range(0, spot_interval_num):
                    IOU_list.append(calculate_IOU(gt_interval_list[i], spot_interval_list[j]))
                if len(IOU_list) > 0 and max(IOU_list) >= 0.5:
                    tp = tp + 1

                    max_index = IOU_list.index(max(IOU_list))
                    tp_index.append(max_index)
                    df = pd.DataFrame([[gt_interval_list[i][0], gt_interval_list[i][1], spot_interval_list[max_index][0], spot_interval_list[max_index][1], max(IOU_list), 'TP']], index=current_index, columns=['T_1', 'T_2', 'P_1', 'P_2', 'K', 'class'])
                    df["T_1"] = df["T_1"].astype("int")
                    df["T_2"] = df["T_2"].astype("int")
                    df["P_1"] = df["P_1"].astype("int")
                    df["P_2"] = df["P_2"].astype("int")
                    if DF is None:
                        DF = df
                    else:
                        DF = pd.concat([DF, df], axis=0)
                else:
                    df = pd.DataFrame([[gt_interval_list[i][0], gt_interval_list[i][1], -1, -1, 0, 'FN']], index=current_index, columns=['T_1', 'T_2', 'P_1', 'P_2', 'K', 'class'])
                    df["T_1"] = df["T_1"].astype("int")
                    df["T_2"] = df["T_2"].astype("int")
                    df["P_1"] = df["P_1"].astype("int")
                    df["P_2"] = df["P_2"].astype("int")
                    if DF is None:
                        DF = df
                    else:
                        DF = pd.concat([DF, df], axis=0)

            TP.append(tp)
            for spot_index in range(0, spot_interval_num):
                if spot_index not in tp_index:
                    df = pd.DataFrame([[-1, -1, spot_interval_list[spot_index][0], spot_interval_list[spot_index][1], 0, 'FP']], index=current_index, columns=['T_1', 'T_2', 'P_1', 'P_2', 'K', 'class'])
                    df["T_1"] = df["T_1"].astype("int")
                    df["T_2"] = df["T_2"].astype("int")
                    df["P_1"] = df["P_1"].astype("int")
                    df["P_2"] = df["P_2"].astype("int")
                    if DF is None:
                        DF = df
                    else:
                        DF = pd.concat([DF, df], axis=0)


    print(TP)
    TP_num = sum(TP)
    print('TP_num : ', TP_num)
    FP_num = sum(SPOT_num) - TP_num
    FN_num = sum(GT_num) - TP_num
    print('FP_num : ', FP_num)
    print('FN_num : ', FN_num)
    if (TP_num + FP_num) == 0:
        Precision = 0
    else:
        Precision = TP_num / (TP_num + FP_num)

    if (TP_num + FN_num) == 0:
        Recall = 0
    else:
        Recall = TP_num / (TP_num + FN_num)

    if (TP_num > 0):
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    else:
        F1 = 0
    print(" Precision - Recall - F1 ", Precision, Recall, F1)

    # save spotting results
    rst_save_path = os.getcwd() + '/'+ 'spotting_results_record_1.xlsx'
    DF.to_excel(rst_save_path)


def main():
    # calculate fusion results
    Evaluate_CASme2(window_len_1=11, kl_1=4, kr_1=6, peakl_1=3, peakr_1=4, p_1=14,
                    window_len_2=11, kl_2=6, kr_2=4, peakl_2=4, peakr_2=3, p_2=14,
                    window_len_3=11, peaklr_len_3=3, p_3=14,
                    window_len_4=15, kl_4=6, kr_4=8, peakl_4=4, peakr_4=5, p_4=5,
                    window_len_5=15, kl_5=8, kr_5=6, peakl_5=5, peakr_5=4, p_5=5,
                    window_len_6=15, peaklr_len_6=5, p_6=5)


if __name__ == '__main__':
    main()
