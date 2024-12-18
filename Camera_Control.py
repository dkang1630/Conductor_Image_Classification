import serial
import time
from picamera2 import Picamera2, Preview
import cv2

#serial connection
uart = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)
# 
# def test_uart():
#     try:
#         message = "Messasge from RSPI\n"
#         print(f"Sending: {message.strip()}")
#         uart.write(message.encode())
#         
#         time.sleep(1)
#         
#         if uart.in_waiting > 0:
#             response = uart.read(uart.in_waiting)
#             print(f"Received: {response}")
#         else:
#             print("No response received.")
#             
#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         uart.close()
#         
#         
# test_uart()
#initialize Picamera2
picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)

preview_config = picam2.create_preview_configuration(main={"size": (1920,1080)})
picam2.configure(preview_config)
picam2.start()
time.sleep(1)


#Manual focus first mode
picam2.set_controls({"AfMode": 0, "LensPosition": 10.0})
time.sleep(3)
#Capture 1st image
RGB888 = picam2.capture_array("main")
cv2.imwrite("test_1st.png", RGB888)

#Manual focus second img
picam2.set_controls({"AfMode": 0, "LensPosition": 8.6})
time.sleep(3)
#Capture 2nd image
RGB888_2 = picam2.capture_array("main")
cv2.imwrite("test_2nd.png", RGB888_2)
picam2.stop()
print("Finished with camera")

print(f"Sending message...")
uart.write(b'rotate motor')

time.sleep(1)

if uart.in_waiting > 0:
    response = uart.read(uart.in_waiting)
    print(f"Received: {response}")
else:
    print("No response received.")
    
uart.close()
    
