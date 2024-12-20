import time
import sys
from servo import Servo, servo2040
import machine

uart = machine.UART(1, baudrate=9600, tx= 20, rx= 21)

s = Servo(servo2040.SERVO_1)
time.sleep(1)

#initalize variable
motor_running = False

while True:
    if uart.any():
        data = uart.read()
        print("Received")
        
        if data == b'rotate motor' and not motor_running:
            
            s.enable()
            
            motor_running = True 
            SWEEP_STEPS = 1260 #4cm along the wire
            SWEEP_SPEED = 0.001

            for i in range(0, SWEEP_STEPS + 1): #from 0 to 90 degrees
                s.value(i) #set the servo to the calculated position
                time.sleep(SWEEP_SPEED)
            
            s.disable()
            #Sebd message back after motor completes
            response = b"motor ran successfully"
            uart.write(response)
            
            motor_running = False
            time.sleep(1)
            
        time.sleep(0.1)

