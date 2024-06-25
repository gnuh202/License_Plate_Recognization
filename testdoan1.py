import queue 
import serial
from threading import Thread
def serial_reader(ser, data_queue):
    data = ser.readline().hex() # Đọc dữ liệu từ UART và chuyển sang kiểu string
    data_queue.put(data) # Lưu dữ liệu vào hàng đợi
    ser.close()
def main():
    while True:
        if not data_queue.empty():
            data = data_queue.get()
            print(data) 
    
ser = serial.Serial('COM4', 9600) # Khởi tạo đối tượng Serial, với tốc độ truyền 9600bps và địa chỉ cổng kết nối
data_queue = queue.Queue() # Tạo hàng đợi để lưu trữ dữ liệu Serial
serial_thread = Thread(target=serial_reader, args=(ser, data_queue)) # Tạo một luồng để đọc dữ liệu Serial
serial_thread.start() # Bắt đầu thực thi luồng đọc Serial    
main()