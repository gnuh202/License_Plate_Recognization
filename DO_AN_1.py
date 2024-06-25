import datetime
import serial
import queue
import os
from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
import tkinter
import cv2
from PIL import ImageTk, Image
from threading import Thread
import time
from time import sleep
import numpy as np
import torch
from numpy import ascontiguousarray

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device

from LP import crop_n_rotate_LP, character_recog_CNN, fixchar
from src.char_classification.model import CNN_Model
weights = "LP_yolov7.pt"
set_logging()
device = select_device('')
model = attempt_load(weights, map_location=device)  # load FP32 model    

CHAR_CLASSIFICATION_WEIGHTS = 'src/weight.h5'
model_char = CNN_Model(trainable=False).model
model_char.load_weights(CHAR_CLASSIFICATION_WEIGHTS)
Min_char = 0.01            #0.012
Max_char = 0.09             #0.07
folder_path = 'images/'

window = Tk()
window.title("BÃI ĐỖ XE THÔNG MINH")
rtsp = 'rtsp://admin:VKAKQP@192.168.1.6:554/H.264'       #162
video = cv2.VideoCapture(0)
canvas_w = window.winfo_screenwidth()
canvas_h = window.winfo_screenheight() 

canvas = Canvas(window, width = canvas_w, height= canvas_h, bg = "#d9e0fa")
canvas.pack()
canvas.create_line(557,0,557,303, width= 3, fill='#00ff00')        #ac1bfa
canvas.create_line(0,301,canvas_w,301, width= 3, fill='#00ff00')   #ac1bfa
#canvas.create_rectangle(1040, 400, 1260, 460, width=4, outline="green")
tieude = tkinter.Label(window, text=" ĐỒ ÁN 1: BÃI ĐỖ XE THÔNG MINH ", fg="green", bg = "#d9e0fa", font=("Time New Roman", 26))
canvas.create_window(760, 5, anchor=NW, window = tieude)

sothe_lb = tkinter.Label(window, text="Số thẻ: ", fg="#091438", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(600, 180, anchor=NW, window = sothe_lb)

giovao_lb = tkinter.Label(window, text="Giờ vào: ", fg="#091438", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(600, 120, anchor=NW, window = giovao_lb)
bienso_lb = tkinter.Label(window, text="Biển số: ", fg="#091438", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(600, 60, anchor=NW, window = bienso_lb)
luuynd_lb = tkinter.Label(window, text="Lưu ý người dùng: ", fg="#091438", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(600, 240, anchor=NW, window = luuynd_lb)
luuynd = tkinter.Label(window, text="", fg="#ff0000", bg = "#d9e0fa", font=("Tahoma", 21, "italic"))
canvas.create_window(850, 240, anchor=NW, window = luuynd)
#anhbs = tkinter.Label(window, text="", fg="#ed4107", bg = "#d9e0fa", font=("Arial", 20, "italic"))
#canvas.create_window(1000, 204, anchor=NW, window = anhbs)

#xetrongbai = tkinter.Label(window, text="Xe trong bãi: ", fg="#091438", bg = "#d9e0fa", font=("Arial", 22))
#canvas.create_window(800, 260, anchor=NW, window = xetrongbai)
#xetrong = tkinter.Label(window, text="Xe trống: ", fg="#091438", bg = "#d9e0fa", font=("Arial", 22))
#canvas.create_window(800, 310, anchor=NW, window = xetrong)

quatrinhxuli = tkinter.Label(window, text="QUÁ TRÌNH XỬ LÍ:", fg="#091438", bg = "#d9e0fa", font=("Arial", 24))
label0 = canvas.create_window(canvas_w/2, 340, window = quatrinhxuli)

title1 = tkinter.Label(window, text="Ảnh chụp được ", fg="#993399", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(30, 370, anchor=NW, window = title1)
title2 = tkinter.Label(window, text="Phát hiện biển số ", fg="#993399", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(330, 370, anchor=NW, window = title2)
title3 = tkinter.Label(window, text="Tiền xử lí ", fg="#993399", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(630, 370, anchor=NW, window = title3)
title4 = tkinter.Label(window, text="Vẽ đường viền ", fg="#993399", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(930, 370, anchor=NW, window = title4)
title5 = tkinter.Label(window, text="Tách từng kí tự ", fg="#993399", bg = "#d9e0fa", font=("Arial", 22))
canvas.create_window(1230, 370, anchor=NW, window = title5)

note0 = tkinter.Label(window, text="", fg="#ff0000", bg = "#d9e0fa", font=("Arial", 20, "italic"))
canvas.create_window(70, 650, anchor=NW, window = note0)
note1 = tkinter.Label(window, text="", fg="#ff0000", bg = "#d9e0fa", font=("Arial", 20, "italic"))
canvas.create_window(70, 690, anchor=NW, window = note1)
note2 = tkinter.Label(window, text="", fg="#ff0000", bg = "#d9e0fa", font=("Arial", 20, "italic"))
canvas.create_window(70, 730, anchor=NW, window = note2)

canvas.create_polygon(290,385, 310,385, 310,380, 320,390, 310,400, 310,395, 290,395, fill="#993399")
canvas.create_polygon(590,385, 610,385, 610,380, 620,390, 610,400, 610,395, 590,395, fill="#993399")
canvas.create_polygon(890,385, 910,385, 910,380, 920,390, 910,400, 910,395, 890,395, fill="#993399")
canvas.create_polygon(1190,385, 1210,385, 1210,380, 1220,390, 1210,400, 1210,395, 1190,395, fill="#993399")


logo = ImageTk.PhotoImage(image=Image.open('logo.png').resize((100, 100)))
canvas.create_image(1400, 10, image = logo, anchor=tkinter.NW)  
#---------------------------------------------------


temp = 0
photo = None
bt1_laythe = 0
image = None
#img01 = None
img02 = None
img03 = None
img04 = None
img05 = None
values = []
def tratt_bt():
    new_window = tkinter.Toplevel()
    new_window.geometry("700x800")
    new_window.title('Tra thông tin biển số')
    canvas_n = Canvas(new_window, width = 700, height= 800, bg = "#d9e0fa")
    canvas_n.pack()
    
    moinhap_lb = tkinter.Label(new_window, text="Mời nhập vào ô tìm kiếm: ", fg="#ff0000", bg = "#d9e0fa", font=("Time New Roman", 15, "italic"))
    moinhap_lb.pack() 
    canvas_n.create_window(10, 10, anchor=NW, window = moinhap_lb)
    
    entry = tkinter.Entry(new_window, width=40)
    entry.pack()
    canvas_n.create_window(10, 48, anchor=NW, window = entry)
    image_objects = []
    def show_text():
        folder_path = 'images/'
        text_entry = entry.get()                  # Lấy dữ liệu từ ô nhập chữ
        if not text_entry:
            messagebox.showinfo("Thông báo", "Bạn chưa nhập thông tin")
            return
        if len(text_entry) < 3:
            messagebox.showinfo("Thông báo", "Cần nhập đủ 3 số trở lên")
            return
        file_list = os.listdir(folder_path)
        image_list = [file_name for file_name in file_list if text_entry in file_name] 
        for obj in image_objects:
            obj["label"].destroy()
            obj["image"].destroy()            
        image_objects.clear()               # Xóa các đối tượng hình ảnh và nhãn trong danh sách
        if len(image_list) == 0:
            messagebox.showinfo("Thông báo", "Không tìm thấy biển số phù hợp")
            return
        for file_i, file_name in enumerate(image_list):
            file_path = os.path.join(folder_path, file_name)
            image_bs = Image.open(file_path)
            photo_bs = ImageTk.PhotoImage(image_bs.resize((270, 200)))

            image_bs = tkinter.Label(new_window, image=photo_bs)
            image_bs.image = photo_bs
            canvas_n.create_image(215, 100+250*file_i, anchor=tkinter.NW, image=image_bs.image)
            
            label_name = 'Mã thẻ: ' + file_name[21:31] + '   Thời gian vào: ' + file_name[9:11] + 'h' + file_name[11:13] + ' ' + file_name[13:15] + '/' + file_name[15:17] + '/' + file_name[17:21] 
            
            file_label = tkinter.Label(new_window, text=label_name, fg="#091438", bg = "#d9e0fa", font=("Arial", 10))
            #file_label.pack()
            canvas_n.create_window(200, 310+250*file_i, anchor=tkinter.NW, window=file_label)
            
            image_objects.append({"image": image_bs, "label": file_label})

    timkiem = tkinter.Button(new_window, text="Tìm kiếm", command=show_text)
    timkiem.pack()    
    canvas_n.create_window(270, 45, anchor=NW, window = timkiem)
def laythe():
    global bt1_laythe
    bt1_laythe = 1
#---------------------------------------------------------------------------------

button = Button(window,text = "Tra thông tin", command = tratt_bt)
button.pack()
canvas.create_window(canvas_w-120, 750, anchor=NW, window = button)
'''
button1 = Button(window,text = "Nhận thẻ", command = laythe)
button1.pack()
canvas.create_window(canvas_w-220, 750, anchor=NW, window = button1)
'''
#---------------------------------------------------------------------------------

def serial_reader(ser, data_queue):
    data = ser.read(1).hex() # Đọc dữ liệu từ UART và chuyển sang kiểu string
    data_queue.put(data) # Lưu dữ liệu vào hàng đợi
    ser.close()
'''    
def serial_reader(ser, data_queue):
    data = ser.readline().hex() # Đọc dữ liệu từ UART và chuyển sang kiểu string
    data_queue.put(data) # Lưu dữ liệu vào hàng đợi
    ser.close()
'''
#---------------------------------------------------------------------------------
def com_serial(plate):
    ser_detect = serial.Serial('COM4', 9600, timeout = 5)                  # Khởi tạo kết nối với cổng COM5   
    ser_detect.write(('A'+plate+'B').encode())                             # Gửi data tới phần cứng
    #start_time = time.time()                                       # Bắt đầu đếm thời gian 
    data = ser_detect.read(5).hex()    
    if len(data) != 10:
        data = ''
    ser_detect.close()                                        # Đóng kết nối
    return data
''' 
def com_serial(plate):
    ser_detect = serial.Serial('COM4', 9600, timeout = 5)                  # Khởi tạo kết nối với cổng COM5   
    ser_detect.write(('A'+plate+'B').encode())                             # Gửi data tới phần cứng
    #start_time = time.time()                                       # Bắt đầu đếm thời gian 
    data = ser_detect.readline().hex()    
    if len(data) != 12:
        data = ''
    ser_detect.close()                                        # Đóng kết nối
    return data  
''' 
#---------------------------------------------------------------------------------
def loctrung(char):
    ind_char = []
    np_char = np.array(char)
    w_mean = np_char.transpose()[2].mean()/2
    for i in range(1,len(np_char)):
        if (np_char[i,0] - np_char[i-1,0]) < w_mean:
            ind_char.append(i)
    np_char = np.delete(np_char, ind_char, axis = 0).tolist()
    return np_char

def detect():
    global model_char, img02, img03, img04, img05, image, model, device
    note0.configure(text = '')
    note1.configure(text = '')
    note2.configure(text = '')
    luuynd.configure(text = '')
    bienso_lb.configure(text = "Biển số:")
    sothe_lb.configure(text = 'Số thẻ:')
    giovao_lb.configure(text = 'Giờ vào:')   

    
    if img02:
        canvas.delete(img02)
        if img03:
            canvas.delete(img03)
            if img04:
                canvas.delete(img04)
                if img05:
                    canvas.delete(img05)
    im0 = image
    img1 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(im0, (270, 200)), cv2.COLOR_BGR2RGB)))
    img01 = canvas.create_image(30, 420, image = img1, anchor=tkinter.NW)                
    #____________________________________________________DETECT PLATE___________________________________________________________


    imgsz = 640
    conf_thres = 0.25
    iou_thres = 0.45
    
    # Initialize
    #set_logging()
    #device = select_device('')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    #model = attempt_load(weights, map_location=device)  # load FP32 model    
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16
                                               
    img = letterbox(im0, imgsz, stride = stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = ascontiguousarray(img)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    #for source, img, im0 in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img, False)[0]

    # Inference
    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img, False)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes = None, agnostic = False)
    im0_copy = im0.copy()
    im1_copy = im0.copy()
    error = 0
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        
        if len(det) == 0:
            note1.configure(text = "Lỗi không nhận diện được biển số!")
            error = 1
        else:    
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{"BS"} {conf:.2f}'
                img_to_save = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                plot_one_box(xyxy, im0, label=label, color=[0,0,255], line_thickness=7)
# Cần sửa@@@@@@@@@@@@@@@@               
            #-----------------------------------------------Hình 2 - Sau khi detect biển số-------------------------------------------     
            img2 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(im0, (270, 200)), cv2.COLOR_BGR2RGB)))
            img02 = canvas.create_image(330, 420, image = img2, anchor=tkinter.NW) 
            #________________________________________________________________________________________________________________________
             
            angle, rotate_thresh, LP_rotated = crop_n_rotate_LP(im0_copy, int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
            if (rotate_thresh is None) or (LP_rotated is None):             #Lỗi do tỉ lệ biển số khoogn phù hợp với thông số đặt ra
                note1.configure(text = "Nhận diện được biển số nhưng tỉ lệ 2 chiều không phù hợp với tỉ lệ thực tế, hoặc biển số quá nghiêng!")
                error = 1
            else:  
                #--------------------------------- Hình 3 - Sau tiền xử lí ----------------------------------------------------------  
                img3 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(rotate_thresh, (270, 200)), cv2.COLOR_BGR2RGB)))
                img03 = canvas.create_image(630, 420, image = img3, anchor=tkinter.NW)
                #____________________________________________________________________________________________________________________
                
                LP_rotated_copy = LP_rotated.copy()
                cont, hier = cv2.findContours(rotate_thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                cont = sorted(cont, key=cv2.contourArea, reverse=True)[:17]
                cv2.drawContours(LP_rotated_copy, cont, -1, (100, 255, 255), 2)
                #--------------------------------- Hình 4 - Sau khi vẽ đường viền bao quanh -----------------------------------------
                img4 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(LP_rotated_copy, (270, 200)), cv2.COLOR_BGR2RGB)))
                img04 = canvas.create_image(930, 420, image = img4, anchor=tkinter.NW)  
                #____________________________________________________________________________________________________________________
                
                char_x = []
                height, width, _ = LP_rotated.shape
                roiarea = height * width
                for ind, cnt in enumerate(cont):
                    (x, y, w, h) = cv2.boundingRect(cont[ind])  
                    ratiochar = w / h
                    char_area = w * h
                    if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                        char_x.append([x, y, w, h])                        
                if not char_x:
                    note1.configure(text = "Không tìm được kí tự nào trên biển số xe do phần tiền xử lí chưa chính xác")                 
                    error = 1
                else:    
                    char_x = np.array(char_x)
                    threshold_12line = char_x[:, 1].min() + (char_x[:, 3].mean() / 2)
                    char_x = sorted(char_x, key=lambda x: x[0], reverse=False)
                    
                    line1 = list(filter(lambda x: x[1] < threshold_12line, char_x))
                    line2 = list(filter(lambda x: x[1] >= threshold_12line, char_x))
                    line1 = loctrung(line1)
                    line2 = loctrung(line2)
                    char_x = line1 + line2
                    char_x = sorted(char_x, key=lambda x: x[0], reverse=False)
                                     
                    strFinalString = ""
                    first_line = ""
                    second_line = ""                         
                    for inde, char in enumerate(char_x):
                        x, y, w, h = char
                        cv2.rectangle(LP_rotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        imgROI = rotate_thresh[y:y + h, x:x + w]
                        text = character_recog_CNN(model_char, imgROI)
                        if text == 'Background':
                            text = ''
                        if y < threshold_12line:
                            
                            first_line += text
                        else:
                            second_line += text
                    #-------------------------------- Kiểm tra nhầm lẫn và sửa kí tự do model gây ra (thủ công) ---------------------------
                    er = ['','','']
                    if (len(first_line) != 4):
                        er[0] = 'Hàng trên không nhận đủ đúng 4 kí tự'
                        #note.configure(text = 'Hàng trên không nhận đủ đúng 4 kí tự')
                        error = 1
                    if ((len(second_line) < 4) or (len(second_line) > 5) ):
                        er[1] = 'Hàng thứ 2 không nhận đủ 4,5 kí tự hoặc dư'
                        #note.configure(text = 'Hàng thứ 2 không nhận đủ 4,5 kí tự hoặc dư')
                        error = 1
                    if len(first_line) < 4:     
                        first_line_fixed, second_line_fixed, bol = first_line, second_line, 0
                    else:
                        first_line_fixed, second_line_fixed, bol = fixchar(first_line, second_line)         
                    if bol:
                        er[2] = 'Nhận diện chữ sai!'
                        #note.configure(text = 'Nhận diện chữ sai!')
                        error = 1 
                    count_er = len([x for x in er if isinstance(x, str) and x != ""])
                    indices = [index for index, value in enumerate(er) if isinstance(value, str) and value != ""]
                    
                    if count_er == 1:
                        note1.configure(text = er[indices[0]])
                    elif count_er == 2:
                        note0.configure(text = er[indices[0]])
                        note1.configure(text = er[indices[1]])    
                    elif count_er == 3:
                        note0.configure(text = er[indices[0]])
                        note1.configure(text = er[indices[1]])   
                        note2.configure(text = er[indices[2]]) 
                    #_____________________________________________________________________________________________________________________   
                    strFinalString = first_line_fixed[:2] + '-' + first_line_fixed[2:] + ' ' + second_line_fixed  
                    bienso_lb.configure(text = "Biển số:  " + str(strFinalString))
                    if len(second_line_fixed) == 4:
                        strFinalString_save = first_line_fixed + second_line_fixed + ' '
                    elif len(second_line_fixed) == 5:
                        strFinalString_save = first_line_fixed + second_line_fixed
                    #---------------------------------- Hình 5 - Khoanh vùng kí tự tìm được ----------------------------------------------
                    img5 = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(cv2.resize(LP_rotated, (270, 200)), cv2.COLOR_BGR2RGB)))
                    img05 = canvas.create_image(1230, 420, image = img5, anchor=tkinter.NW)  
                    #_____________________________________________________________________________________________________________________
    if (error == 1):
        luuynd.configure(text = 'Hệ thống nhận diện lỗi, xin mời bấm nhận thẻ lại!!')          
    else:
        luuynd.configure(text = 'Đã nhận diện xong!')
        sothe_lb.configure(text = 'Số thẻ:  Đang lấy mã thẻ...')                    #'Số thẻ:  Đang lấy mã thẻ...'
        now = datetime.datetime.now()
        giovao_lb.configure(text = 'Giờ vào:  ' + str(now.strftime('%H:%M %d-%m-%Y'))) 
        
        form_name = strFinalString_save + str(now.strftime('%H%M%d%m%Y'))
        datasend = strFinalString_save + str(now.strftime('%Hh%M'))
        sothe = com_serial(datasend)
        if not sothe:
            sothe_lb.configure(text = 'Số thẻ:  Lỗi đọc thẻ')
        else: 
            sothe_lb.configure(text = 'Số thẻ:  ' + sothe.upper())
        cv2.imwrite(folder_path + form_name + sothe + '.jpg', img_to_save)
        
    ser = serial.Serial('COM4', 9600) # Khởi tạo đối tượng Serial, với tốc độ truyền 9600bps và địa chỉ cổng kết nối    
    serial_thread = Thread(target=serial_reader, args=(ser, data_queue)) # Tạo một luồng để đọc dữ liệu Serial
    serial_thread.start() # Bắt đầu thực thi luồng đọc Serial
    sleep(200)
    '''
    note0.configure(text = '')
    note1.configure(text = '')
    note2.configure(text = '')
    luuynd.configure(text = '')
    bienso_lb.configure(text = "Biển số:")
    sothe_lb.configure(text = 'Số thẻ:')
    giovao_lb.configure(text = 'Giờ vào:')
    '''

def update_frame():
    global canvas, photo, bt1_laythe, image
    # Doc tu camera
    ret, image = video.read()
    image = image[0:1000,0:1900]
    #frame = cv2.flip(frame, 1)
    #frame = cv2.resize(frame, dsize=None, fx=0.5, fy=0.5) 
    frame = cv2.resize(image, (556, 300)) 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    photo = ImageTk.PhotoImage(image=Image.fromarray(frame)) 
    canvas.create_image(0,0, image = photo, anchor=tkinter.NW) 
    
    if not data_queue.empty():
        data = data_queue.get()
        print(data)
        if data == '40':
            t1 = Thread(target = detect)
            t1.start()
        else:
            ser = serial.Serial('COM4', 9600) # Khởi tạo đối tượng Serial, với tốc độ truyền 9600bps và địa chỉ cổng kết nối    
            serial_thread = Thread(target=serial_reader, args=(ser, data_queue)) # Tạo một luồng để đọc dữ liệu Serial
            serial_thread.start() # Bắt đầu thực thi luồng đọc Serial
            
    '''
    if (bt1_laythe == 1):
        bt1_laythe = 0
        t1 = Thread(target = detect)
        t1.start()
    '''   
    window.after(10, update_frame)

ser = serial.Serial('COM4', 9600) # Khởi tạo đối tượng Serial, với tốc độ truyền 9600bps và địa chỉ cổng kết nối
data_queue = queue.Queue() # Tạo hàng đợi để lưu trữ dữ liệu Serial
serial_thread = Thread(target=serial_reader, args=(ser, data_queue)) # Tạo một luồng để đọc dữ liệu Serial
serial_thread.start() # Bắt đầu thực thi luồng đọc Serial
update_frame()
window.mainloop()


