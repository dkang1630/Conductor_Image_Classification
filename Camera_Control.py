import serial
import time
from picamera2 import Picamera2, Preview
import cv2
import os

#serial connection
uart = serial.Serial('/dev/serial0', baudrate=9600, timeout=1)

#Image folder path
image_folder = "/home/VibRoLab/MDR_img"

#Get next avaiable image number
def get_next_image_number():
    existing_files = os.listdir(image_folder)
    image_numbers = []
    
    #Check for existing image files and extract numbers
    for files in existing_files:
        try:
            split_name = files.split('_')
            if len(split_name) > 1:
                try:
                    number_str = split_name[0] #number
                    number = int(number_str)
                    image_numbers.append(number)
                except ValueError:
                    print(f"Skipping file {files}: number extraction failed.")
            else:
                print(f"Skipping file {files}: Filename format doesn't match expected pattern.")
        except Exception as e:
            print(f"Error processing file {file}: {e}")
        
    if not image_numbers:
        return 1
    return max(image_numbers) + 1

#Number of times to run the system
num_iterations = 30 #around 100? cm in totoal

for i in range (num_iterations):
    print(f"Iteration {i + 1} of {num_iterations}")

    picam2 = Picamera2()
    time.sleep(1)
    picam2.start_preview(Preview.QTGL)

    preview_config = picam2.create_preview_configuration(main={"size": (1920,1080)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(1)
    
    #Manual focus first mode
    picam2.set_controls({"AfMode": 0, "LensPosition": 10.0})
    time.sleep(3)
    #Capture 1st image
    first_image_number = get_next_image_number()
    RGB888 = picam2.capture_array("main")
    first_image_path = os.path.join(image_folder, f"{first_image_number}_img.jpg")
    cv2.imwrite(first_image_path, RGB888)
    print("Saved first image")

    #Manual focus second img
    picam2.set_controls({"AfMode": 0, "LensPosition": 8.2})
    time.sleep(3)
    #Capture 2nd image
    second_image_number = get_next_image_number()
    RGB888_2 = picam2.capture_array("main")
    second_image_path = os.path.join(image_folder, f"{second_image_number}_img.jpg")
    cv2.imwrite(second_image_path, RGB888_2)
    print("Saved second image")
    
    #stop camera
    picam2.stop()
    picam2.close()
    print("Finished with camera")

    print(f"Sending message... Iterations {i+1}")
    uart.write(b'rotate motor')
    
    #wait for the resonse
    while uart.in_waiting == 0:
        time.sleep(0.1)
    

    response = uart.read(uart.in_waiting)
    print(f"Received: {response}")
    
    time.sleep(1)
    
uart.close()
print("All iterations completed")    
