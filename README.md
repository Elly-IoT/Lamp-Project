# Under Development
Still under Development

# Lamp-Project
A project where we used Tensorflow Lite on a Raspberry Pi to detect a Book and control a camera on the RPI to follow the book.

# Preperations
To prepare your raspberry pi to run Tensorflow and all the other stuff we need to have a couple of things in place 

If you havent done yet - donwload the raspberry pi OS - I am using the May 2020 Version with recommended software (Raspberry Pi OS (32-bit) with desktop and recommended software)

Use Balena Etcher or any other imaging Tool to flash the OS to the SD Card

```python

# Upgrade Raspberry Pi
sudo apt-get update
sudo apt-get dist-upgrade

# Clone this repo to your raspberry pi
# https://github.com/Elly-IoT/Lamp-Project.git

# Make sure camera is on
# enable the i2c and camera interfaces via the menu
sudo raspi-config


# Create a virtual environment using venv
python3 -m venv raspilamp-env

# Activate the environment using (remember you might need to activate it again)
source raspilamp-env/bin/activate
# (raspilamp-env) pi@raspberrypi:~ $ This is how your command line should look now

# Get packages required for OpenCV
sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev

# Need to get an older version of OpenCV because version 4 has errors
pip3 install opencv-python==3.4.6.27

# Get packages required for TensorFlow
# Using the tflite_runtime packages available at https://www.tensorflow.org/lite/guide/python
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

# With all of this done you should now be able to run the Tensorflow lite model with a webcam 
python3 TFLite_detection_webcam.py --modeldir=Sample_TFLite_model

# SymLink smbus http://www.netzmafia.de/skripten/hardware/RasPi/RasPi_I2C.html
# This means we are not downloading a dependency but rather creating a link to a package which is already there for this we need to use the ln command within the site packages folder 
cd raspilamp-env/lib/python3.7/site-packages/
ln -s /usr/lib/python3/dist-packages/smbus.cpython-37m-arm-linux-gnueabihf.so smbus.so

# Install possibly missing packages
pip install pantilthat
pip install imutils
pip install "picamera[array]"


#Afterwards you should be able to run 
python pan_tilt_tracking.py --modeldir=coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/
#At this point this will still only open the webcam but later on this will 




```
