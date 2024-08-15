import os
import xlrd
import numpy as np


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
    workpath = r'D:\ME_Database_Download\CAS(ME)^2\CAS(ME)^2code_final.xlsx'  # path of CAS(ME)^2 label file
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


def get_spotted_intervals_for_video(video_name, window_len, peaklr_len, p):
    contrast_differences_path = r'..\..\..\results\contrast differences\CAS(ME)^2_LBP_contrast_differences' + '\\' + 'window_len_' + str(window_len)
    workpath = contrast_differences_path + '\\' + 's' + video_name[:2] + '\\' + video_name[:-4] + '\\' + 'contrast_differences.xls'

    data = xlrd.open_workbook(workpath)
    data_v = data.sheet_by_name('sheet1')
    rowNum_dv = data_v.nrows
    Carr = []
    for i in range(1, rowNum_dv):
        Carr.append(data_v.cell_value(i, 2))

    Cmean = np.sum(Carr[int(window_len/2)*2:len(Carr)-int(window_len/2)*2])/len(Carr)
    Cmax = np.max(Carr[int(window_len/2)*2:len(Carr)-int(window_len/2)*2])
    epsilon = 0.01 * p
    # threshold
    Thr = Cmean + epsilon * (Cmax - Cmean)

    res = np.zeros(len(Carr))
    for i in range(int(window_len/2)*2, len(Carr)-int(window_len/2)*2):
        if (Carr[i] >= Thr):
            res[i] = 1
            j = i + 1
            k = i - 1
            count = 0
            while (count <= peaklr_len - 1):
                count = count + 1
                if j <= len(Carr) - 1:
                    res[j] = 1
                if k >= 0:
                    res[k] = 1
                j = j + 1
                k = k - 1
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


def Evaluate_CASme2(window_len, peaklr_len, p):
    folder_data = r'D:\ME_Database_Download\CAS(ME)^2\rawvideo'   # path of CAS(ME)^2 dataset
    subfolders = os.listdir(folder_data)
    TP = []
    GT_num = []
    SPOT_num = []
    for sub_folder in subfolders:
        vidfolders = os.listdir(os.path.join(folder_data, sub_folder))
        for vidname in vidfolders:
            tp = 0
            print(vidname)

            sub_idx = int(sub_folder[-2:])
            spot_interval_list = get_spotted_intervals_for_video(vidname, window_len, peaklr_len, p)
            gt_interval_list = get_GT_intervals_for_video(sub_idx, vidname)

            gt_interval_num = len(gt_interval_list)
            spot_interval_num = len(spot_interval_list)
            print(gt_interval_num)
            print(spot_interval_num)
            GT_num.append(gt_interval_num)
            SPOT_num.append(spot_interval_num)

            for i in range(0, gt_interval_num):
                IOU_list = []
                for j in range(0, spot_interval_num):
                    IOU_list.append(calculate_IOU(gt_interval_list[i], spot_interval_list[j]))
                if len(IOU_list) > 0 and max(IOU_list) >= 0.5:
                    tp = tp + 1

            TP.append(tp)

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


def main():
    Evaluate_CASme2(window_len=11, peaklr_len=3, p=3)
    # Evaluate_CASme2(window_len=15, peaklr_len=5, p=6)


if __name__ == '__main__':
    main()
