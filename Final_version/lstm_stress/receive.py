from pyrf24 import RF24, RF24_PA_MAX, RF24_250KBPS
import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed_time(self):
        if self.start_time is None:
            return None
        elif self.end_time is None:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time

timer = Timer()
timer.start()

# Set up nRF24L01 radio on SPI bus
radio = RF24(22, 0)
radio.begin()
radio.setChannel(115)
radio.setPALevel(RF24_PA_MAX)
radio.setDataRate(RF24_250KBPS)
radio.openReadingPipe(0,0x0000000001)
radio.startListening()

# Loop forever
while True:
    
    if timer.elapsed_time()<0.2:
        
        continue

    timer.start()
    
    # Check if data is available
    if radio.available():
        # Read the data
        data = radio.read(32)
        
        id_node = data[0]
        id_bt = data[1]
        print(f"{id_node}, {id_bt}")
        print("Received data: ", data)
