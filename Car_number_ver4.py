# 이미지 크기 비율 고정 조정
# 텍스트 를 긁어온 부분만 보기
# 정규식 을 활용해
# xx노xxxx
# xxx노xxxx
# 형식의 텍스트가 나오는 이미지만 남기고 나머지는 continue로 다음번째 반복으로 넘어가기.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from pytesseract import image_to_boxes

import re
flag = 0

def find_chars(contour_list):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        # recursive
        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


cnt_match = 0


plt.style.use('dark_background')

# 이미지를 읽어와서 리사이즈.
# img = cv2.imread('./car/pic_008.jpg')
img = cv2.imread('./images/car5.jpeg')


if img is None:
    print("이미지가 로드되지 않았거나 비어 있습니다.")
height, width, channel = img.shape

# 이미지의 최소 너비와 최대 너비를 정의합니다.
min_width, max_width = 200, 2401

# 원본 이미지의 가로 비율 계산
original_ratio = width / height

# 이미지 크기를 조절하는 for문
for i in range(min_width, max_width + 1):
    # 비율을 유지하면서 조절
    flag = 0
    j = int(i / original_ratio)

    # 이미지 크기 조정
    resized_img = cv2.resize(img, (i, j))

    # cv2.cvtColor 의 cv2.COLOR_BGR2GRAY 을 사용해 GRAY색상으로 변경.
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # 잡음 제거 GaussianBlur 블러 적용.
    GaussianBlur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    th = cv2.adaptiveThreshold(GaussianBlur_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 19, 9)

    # max value 는 최대값을 나타냄.
    # 임계값을 넘으면 지정된 최대값으로 설정하는데, 여기서는 255로 설정되었습니다.
    # 이 값은 흑백 이미지에서 사용되며, 최대 밝기를 나타냅니다.

    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C: 적응형 임계값 처리 방법을 나타냅니다.
    # 여기서는 가우시안 평균을 사용하도록 설정되었습니다.

    # blockSize 블록 크기입니다. 가우시안 평균을 계산할 때 사용되는 이웃 픽셀의 크기를 나타냅니다.
    # 이 값은 홀수여야 합니다.

    # C, 2: 가중치 매개변수 C입니다. 평균에서 뺄 값을 나타냅니다. 높은 값은 더 적은 픽셀을 흰색으로 만듭니다.
    # print(img.shape)  # 이미지의 크기 출력

    contours, _ = cv2.findContours(
        th,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))
    # 윤곽선 그리기.
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    # 어떤게 번호판처럼 생겼는지?

    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

    # 리얼 번호판 추려내기
    MAX_DIAG_MULTIPLYER = 5  # 5
    MAX_ANGLE_DIFF = 12.0  # 12.0
    MAX_AREA_DIFF = 0.5  # 0.5
    MAX_WIDTH_DIFF = 0.8
    MAX_HEIGHT_DIFF = 0.2
    MIN_N_MATCHED = 3  # 3

    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                          thickness=2)

    # 똑바로 돌리기
    PLATE_WIDTH_PADDING = 1.3  # 1.3
    PLATE_HEIGHT_PADDING = 1.5  # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )

        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

        img_rotated = cv2.warpAffine(th, M=rotation_matrix, dsize=(width, height))

        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[
            0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

    # 최종 확인
    longest_idx, longest_text = -1, 0
    plate_chars = []

    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 다시 contours 찾기 (위와 동일)
        contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            area = w * h
            ratio = w / h

            if area > MIN_AREA \
                    and w > MIN_WIDTH and h > MIN_HEIGHT \
                    and MIN_RATIO < ratio < MAX_RATIO:
                if x < plate_min_x:
                    plate_min_x = x
                if y < plate_min_y:
                    plate_min_y = y
                if x + w > plate_max_x:
                    plate_max_x = x + w
                if y + h > plate_max_y:
                    plate_max_y = y + h

        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT,
                                        value=(0, 0, 0))



        # 문자 경계 상자 정보 가져오기
        boxes = image_to_boxes(img_result, lang='kor')

        for b in boxes.splitlines():
            b = b.split()
            x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(img_result, (x, img_result.shape[0] - y), (w, img_result.shape[0] - h), (0, 255, 0), 2)

        chars = pytesseract.image_to_string(img_result, lang='kor_k', config='--psm 7 --oem 1')

        result_chars = ''
        has_digit = False
        for c in chars:
            if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
                if c.isdigit():
                    has_digit = True
                result_chars += c

        pattern = r"\d{2,3}[가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자배하허호]{1}\d{4}"  # 정규 표현식 패턴
        match = re.search(pattern, result_chars)  # 문자열에서 패턴 찾기




        if match:
            result = match.group()  # 찾은 패턴 반환
            print(result)
        else:
            # cnt_match += 1
            # if cnt_match == 1000:
            #     cnt_match = 0
            # print(f'일치 하는 패턴 없음. \ncount{cnt_match}')
            # print(f'Size: {plate_img.shape[1]} x {plate_img.shape[0]}')
            continue
        plate_chars.append(result_chars)

        if has_digit and len(result_chars) > longest_text:
            longest_idx = i

        plt.subplot(len(plate_imgs), 1, i + 1)

        # 이미지 크기 정보 표시
        plt.title(f"Result Image {i + 1} - Size: {plate_img.shape[1]} x {plate_img.shape[0]}")
        plt.imshow(img_result, cmap='gray')
        plt.show()
        # if result_chars == '30라1451':
        #     # plt.show()
        #     flag = 1
        #     break
            # sys.exit()  # 프로그램 종료
            # 결과가 입력으로 준 문자열인 경우 plt.show() 실행.
            # 닫으면 사이즈 조절 for문 종료
    if flag == 1:
        break

