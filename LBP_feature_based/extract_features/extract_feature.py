import dlib
import cv2
import os
import lbp
import xlsxwriter
from imutils import face_utils


p68 = "/landmark_model/shape_predictor_68_face_landmarks.dat" 
detector = dlib.get_frontal_face_detector()
predictor_68 = dlib.shape_predictor(p68)


def divide_image_to_block(gray_img):
    pos_x = [[1, 37], [38, 74], [75, 111], [112, 148], [149, 185], [186, 227]]
    pos_y = [[1, 37], [38, 74], [75, 111], [112, 148], [149, 185], [186, 227]]

    list_img_block = []
    for xa in pos_x:
        for ya in pos_y:
            img_block = gray_img[ya[0]:ya[1], xa[0]: xa[1]]
            list_img_block.append(img_block)

    return list_img_block


def get_facial_region(image):
    dets = detector(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 2)
    print(dets)
    if (len(dets) > 0):
        iface = 0
        face_distance = 0
        for (j, d) in enumerate(dets):
            if (d.right() - d.left() > face_distance):
                face_distance = d.right() - d.left()
                iface = j
        detected_face = dets[iface]

    shape68 = predictor_68(image, detected_face)
    shape = face_utils.shape_to_np(shape68)
    Ll = shape[0]
    Lr = shape[0]
    Lt = shape[0]
    Lb = shape[0]
    for i in range(1, 68):
        if shape[i][0] < Ll[0]:
            Ll = shape[i]
        if shape[i][0] > Lr[0]:
            Lr = shape[i]
        if shape[i][1] < Lt[1]:
            Lt = shape[i]
        if shape[i][1] > Lb[1]:
            Lb = shape[i]

    A = [Ll[0], Lt[1] - (shape[36][1] - shape[18][1])]
    B = [Lr[0], Lb[1]]
    boxB = [A, B]

    cropped_image = image[A[1]:B[1], A[0]:B[0]]
    dets_crop = detector(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY), 2)
    print(dets_crop)
    if (len(dets_crop) > 0):
        iface = 0
        face_distance = 0
        for (j, d) in enumerate(dets_crop):
            if (d.right() - d.left() > face_distance):
                face_distance = d.right() - d.left()
                iface = j
        detected_face_crop = dets_crop[iface]

    shape68_crop = predictor_68(cropped_image, detected_face_crop)
    shape_crop = face_utils.shape_to_np(shape68_crop)
    print(shape_crop)
    Lb1 = shape_crop[0]
    for i in range(1, 68):
        if shape_crop[i][1] > Lb1[1]:
            Lb1 = shape_crop[i]
    print(Lb1)
    B1 = [Lr[0], min(Lb[1], Lb1[1] + A[1])]
    boxB1 = [A, B1]
    return boxB1


def extract_LBP_features_for_video(video_file):
    save_path = "/results/features/CAS(ME)^2_LBP_features" + '/' + video_file.split('/')[-2] "
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print("Starting !!")
    cap = cv2.VideoCapture(video_file)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Number of frames in this video: ', num_frame)

    idx = 0
    features = []
    while (cap.isOpened()):
        ret, img = cap.read()
        if (ret == False):
            break
        idx = idx + 1
        if (idx - 1 == 0):
            box = get_facial_region(img)

        cropped_img = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        resized_img = cv2.resize(cropped_img, (227, 227))
        lbp_feat_img = lbp.extract_LBP_feature(resized_img)

        blocks_img = divide_image_to_block(resized_img)
        num_block = len(blocks_img)
        print('Number of blocks: ', num_block)
        lbp_feat_blocks = []
        for iblock in range(0, num_block):
            blck_img = blocks_img[iblock]
            lbp_feat_block = lbp.extract_LBP_feature(blck_img)
            lbp_feat_blocks.append(lbp_feat_block)

        feature = [lbp_feat_img, lbp_feat_blocks]
        features.append(feature)
        print('Have processed', idx, 'frames.')
    cap.release()

    # save results, 37 sheets
   save_file = os.path.join(save_path, video_file.split('/')[-1][:-4] + '_features.xls')
    workbook = xlsxwriter.Workbook(save_file)
    for i_sheet in range(0, 37):
        worksheet = workbook.add_worksheet('sheet'+str(i_sheet))
        worksheet.write(0, 0, '#frames')
        for i_row in range(1, num_frame + 1):
            worksheet.write(i_row, 0, i_row)
            if i_sheet == 0:
                feature_lbp = features[i_row - 1][0]
            else:
                feature_lbp = features[i_row - 1][1][i_sheet-1]
            for i_col in range(1, len(feature_lbp) + 1):
                print(i_col)
                worksheet.write(i_row, i_col, float(feature_lbp[i_col - 1][0]))
    workbook.close()

    print('Saving...')


def extract_LBP_features_for_sequence(video_file):
    save_path = "/results/features/SAMM_Long_Videos_LBP_features"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    print("Starting !!")
    images = os.listdir(video_file)
    num_frame = len(images)
    print('Number of frames in this video: ', num_frame)

    features = []
    for i in range(0, num_frame):
        img_name = images[i]
        img = cv2.imread(os.path.join(video_file, img_name))
        idx = i + 1
        if (idx - 1 == 0):
            box = get_facial_region(img)

        cropped_img = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        resized_img = cv2.resize(cropped_img, (227, 227))
        lbp_feat_img = lbp.extract_LBP_feature(resized_img)

        blocks_img = divide_image_to_block(resized_img)
        num_block = len(blocks_img)
        print('Number of blocks: ', num_block)
        lbp_feat_blocks = []
        for iblock in range(0, num_block):
            blck_img = blocks_img[iblock]
            lbp_feat_block = lbp.extract_LBP_feature(blck_img)
            lbp_feat_blocks.append(lbp_feat_block)

        feature = [lbp_feat_img, lbp_feat_blocks]
        features.append(feature)
        print('Have processed', idx, 'frames.')

    # save results, 37 sheets
    save_file = os.path.join(save_path, video_file.split('/')[-1] + '_features.xls')
    workbook = xlsxwriter.Workbook(save_file)
    for i_sheet in range(0, 37):
        worksheet = workbook.add_worksheet('sheet' + str(i_sheet))
        worksheet.write(0, 0, '#frames')
        for i_row in range(1, num_frame + 1):
            worksheet.write(i_row, 0, i_row)
            if i_sheet == 0:
                feature_lbp = features[i_row - 1][0]
            else:
                feature_lbp = features[i_row - 1][1][i_sheet - 1]
            for i_col in range(1, len(feature_lbp) + 1):
                worksheet.write(i_row, i_col, float(feature_lbp[i_col - 1][0]))
    workbook.close()

    print('Saving...')


def save_LBP_features_for_CASme2():
    folder_data = " "   # path of CAS(ME)^2 dataset
    subfolders = os.listdir(folder_data)

    for sub_folder in subfolders:
        vidfolders = os.listdir(os.path.join(folder_data, sub_folder))
        for vidname in vidfolders:
            print(vidname)
            extract_LBP_features_for_video(os.path.join(folder_data, sub_folder, vidname))

    print('Successfully saved.')


def save_LBP_features_for_SAMM_Long_Videos():
    folder_data = " "  # path of SAMM Long Videos dataset
    vidfolders = os.listdir(folder_data)

    for vidname in vidfolders:
        print(vidname)
        extract_LBP_features_for_sequence(os.path.join(folder_data, vidname))

    print('Successfully saved.')


def main():
    save_LBP_features_for_CASme2()
    save_LBP_features_for_SAMM_Long_Videos()


if __name__ == '__main__':
    main()
