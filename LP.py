import cv2
import math
import numpy as np

GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)  # kích cỡ càng to thì càng mờ
ADAPTIVE_THRESH_BLOCK_SIZE = 19
ADAPTIVE_THRESH_WEIGHT = 9
COLORS = np.random.uniform(0, 255, size=(1, 3))

ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2',
              24: '3', 25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}

def extractValue(imgOriginal):
    height, width, numChannels = imgOriginal.shape
    imgHSV = np.zeros((height, width, 3), np.uint8)
    imgHSV = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2HSV)
    imgHue, imgSaturation, imgValue = cv2.split(imgHSV)
    return imgValue

def maximizeContrast(imgGrayscale):             # Làm cho độ tương phản lớn nhất
    height, width = imgGrayscale.shape
    imgTopHat = np.zeros((height, width, 1), np.uint8)
    imgBlackHat = np.zeros((height, width, 1), np.uint8)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # tạo bộ lọc kernel
    imgTopHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_TOPHAT, structuringElement, iterations=10)  # nổi bật chi tiết sáng trong nền tối
    imgBlackHat = cv2.morphologyEx(imgGrayscale, cv2.MORPH_BLACKHAT, structuringElement, iterations=10)  # Nổi bật chi tiết tối trong nền sáng
    imgGrayscalePlusTopHat = cv2.add(imgGrayscale, imgTopHat)
    imgGrayscalePlusTopHatMinusBlackHat = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    return imgGrayscalePlusTopHatMinusBlackHat

def preprocess(imgOriginal):
    imgGrayscale = extractValue(imgOriginal)
    imgMaxContrastGrayscale = maximizeContrast(imgGrayscale)  # để làm nổi bật biển số hơn, dễ tách khỏi nền
    height, width = imgGrayscale.shape
    imgBlurred = np.zeros((height, width, 1), np.uint8)
    imgBlurred = cv2.GaussianBlur(imgMaxContrastGrayscale, GAUSSIAN_SMOOTH_FILTER_SIZE, 0)
    # Làm mịn ảnh bằng bộ lọc Gauss 5x5, sigma = 0
    imgThresh = cv2.adaptiveThreshold(imgBlurred, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT)
    return imgGrayscale, imgThresh
def rotation_angle(linesP):     #Tính toán góc 
    angles = []
    for i in range(0, len(linesP)):
        l = linesP[i][0].astype(int)
        p1 = (l[0], l[1])
        p2 = (l[2], l[3])
        doi = (l[1] - l[3])
        ke = abs(l[0] - l[2])
        angle = math.atan(doi / ke) * (180.0 / math.pi)
        if abs(angle) > 45:  # Nếu thấy đường kẻ 
            angle = (90 - abs(angle)) * angle / abs(angle)
        angles.append(angle)
    angles = list(filter(lambda x: (abs(x > 3) and abs(x < 15)), angles))
    if not angles:  
        angles = list([0])
    angle = np.array(angles).mean()
    return angle

def rotate_LP(img, angle):      #Xoay, calib biển số lại

    height, width = img.shape[:2]
    ptPlateCenter = width / 2, height / 2
    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
    rotated_img = cv2.warpAffine(img, rotationMatrix, (width, height))
    return rotated_img

def Hough_transform(threshold_image, nol=6):
    h, w = threshold_image.shape[:2]
    linesP = cv2.HoughLinesP(threshold_image, 1, np.pi / 180, 50, None, 50, 10)
    dist = []
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        d = math.sqrt((l[0] - l[2]) ** 2 + (l[1] - l[3]) ** 2)
        if d < 0.5 * max(h, w):
            d = 0
        dist.append(d)
        # cv2.line(threshold_image, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
    dist = np.array(dist).reshape(-1, 1, 1)
    linesP = np.concatenate([linesP, dist], axis=2)
    linesP = sorted(linesP, key=lambda x: x[0][-1], reverse=True)[:nol]

    return linesP

def character_recog_CNN(model, img, dict=ALPHA_DICT):
    imgROI = cv2.resize(img, (28, 28), cv2.INTER_AREA)
    imgROI = imgROI.reshape((28, 28, 1))
    imgROI = np.array(imgROI)
    imgROI = np.expand_dims(imgROI, axis=0)
    result = model.predict(imgROI, verbose='2')
    result_idx = np.argmax(result, axis=1)
    return ALPHA_DICT[result_idx[0]]

def crop_n_rotate_LP(source_img, x1, y1, x2, y2):
    w = int(x2 - x1)
    h = int(y2 - y1)
    ratio = w / h
    if 0.9 <= ratio <= 2.5 or 3.5 <= ratio <= 6.5:
        cropped_LP = source_img[y1:y1 + h, x1:x1 + w]
        
        gray = cv2.cvtColor(cropped_LP, cv2.COLOR_BGR2GRAY)
        bl = cv2.GaussianBlur(gray, (5,5), 0)
        ret1, imgThreshplate = cv2.threshold(bl, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
        
        #imgGrayscaleplate, imgThreshplate = preprocess(cropped_LP)
        canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Canny Edge
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=2)
        linesP = Hough_transform(dilated_image, nol=6)
        for i in range(0, len(linesP)):
            l = linesP[i][0].astype(int)
            # cv2.line(cropped_LP_copy, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
        angle = rotation_angle(linesP)
        rotate_thresh = rotate_LP(imgThreshplate, angle)
        LP_rotated = rotate_LP(cropped_LP, angle)
    else:
        angle, rotate_thresh, LP_rotated = None, None, None
    return angle, rotate_thresh, LP_rotated


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, classes, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect(image, classes, weights, config):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    #classes = None
    with open(classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    net = cv2.dnn.readNet(weights, config)
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    # Thực hiện xác định bằng HOG và SVM
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return boxes, indices, class_ids, confidences, Width, Height

def fixchar(first_line, second_line):   
    bol = 0
    if 'S' in first_line[:2]:
        first_line = first_line[:2].replace('S','5') + first_line[2:]
    if 'D' in first_line[:2]:
        first_line = first_line[:2].replace('D','0') + first_line[2:]
    if 'B' in first_line[:2]:
        first_line = first_line[:2].replace('B','8') + first_line[2:]
    if 'Z' in first_line[:2]:
        first_line = first_line[:2].replace('Z','2') + first_line[2:]
    
    if '5' in first_line[2]:
        first_line = first_line[:2] + 'S' + first_line[3:]
    if '0' in first_line[2]:
        first_line = first_line[:2] + 'D' + first_line[3:]
    if '8' in first_line[2]:
        first_line = first_line[:2] + 'B' + first_line[3:]
    if '2' in first_line[2]:
        first_line = first_line[:2] + 'Z' + first_line[3:]

    if 'S' in second_line:
        second_line = second_line.replace('S','5')
    if 'D' in second_line:
        second_line = second_line.replace('D','0')
    if 'B' in second_line:
        second_line = second_line.replace('B','8')
    if 'Z' in second_line:
        second_line = second_line.replace('Z','2')
        
    for char1 in first_line[:2]:
        if char1.isalpha():
            bol = 1
            break 
   
    if first_line[2].isnumeric():
        bol = 1

    for char0 in second_line:
        if char0.isalpha():
            bol = 1
            break 
    return first_line, second_line, bol 