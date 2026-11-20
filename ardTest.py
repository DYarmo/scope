import serial


def send_startup_parameters(VAL1=None,VAL2=None, port=None, baudrate=None):
    if port is None:
        port = '/dev/ttyACM0'
    if baudrate is None:
        baudrate = 115200
    if VAL1 is None:
        VAL1 = 80
    if VAL2 is None:
        VAL2 = 20
    timeout = 1
    print(f"Opening port {port}...")
    with serial.Serial(port, baudrate, timeout=timeout) as ser:
        # 1. Wait for Arduino to boot and send READY signal
        print("Waiting for Arduino to reset...")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if "READY" in line:
                print("Arduino is ready.")
                break
        
        # 2. Format and send the packet: <val1,val2>
        packet = f"<{VAL1},{VAL2}>\n"
        print(f"Sending parameters: {packet.strip()}")
        ser.write(packet.encode('utf-8'))
        
        # 3. Verify confirmation from Arduino
        print("Waiting for verification...")
        while True:
            line = ser.readline().decode('utf-8').strip()
            if "CONFIRMED" in line:
                print(f"Success! Arduino confirmed receipt: {line}")
                return True
            if "ERROR" in line:
                print("Arduino reported a data error.")
                return False

if __name__ == "__main__":
    send_startup_parameters()
    print("PC task finished. Communication closed safely.")