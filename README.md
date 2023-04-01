```
sudo pip3 install picamera

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update

sudo apt-get install libedgetpu1-std

sudo apt-get install python3-pycoral  --yes

pip3 show tflite_runtime

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime

mkdir google-coral && cd google-coral
git clone https://github.com/google-coral/examples-camera --depth 1

cd examples-camera
sh download_models.sh

cd opencv
bash install_requirements.sh

pip3 install pigpio
pip3 install imutils

sudo apt install python3-opencv

sudo python3 -m pip install pyrf24

sudo python3 send.py
```