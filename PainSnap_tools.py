# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:26:38 2025

@author: david.yarmolinsky
"""

from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QVBoxLayout, QWidget
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QPainter
import sys
import cv2
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import os
#from tcam import TCam
#from pygrabber.dshow_graph import FilterGraph
from cv2_enumerate_cameras import enumerate_cameras
import pyqtgraph as pg
import imageio
import time
import base64
from array import array
import pdb
import io
import glob
import serial
from serial.tools import list_ports
from scipy.stats import mode
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK 
import flirpy
from flirpy.camera.lepton import Lepton



def convert_flir(frame, flir_range=[25, 45]):
    frame=frame/100
    frame=frame - 273.15
    frame=frame-flir_range[0]
    frame=frame/(flir_range[1]-flir_range[0])
    frame[frame > 1]=1
    frame[frame < 0]=0
    frame=frame * 255
    frame=frame.astype(np.uint8)
    return (frame)

def figure_to_np(fig):
    with io.BytesIO() as buff:
        fig.savefig(buff, format='raw')
        buff.seek(0)
        data=np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h=fig.canvas.get_width_height()
    im=data.reshape((int(h), int(w), -1))
    return (im)

def list_flir_cameras():
    devs = {}

    for dev in enumerate_cameras():
        name = cam_info_to_ref_str(dev)
        if not (name in devs) and ('PureThermal' in name):
            record = {}
            record['info'] = dev
            record['name'] = name
            devs[name] = record   

    return devs
    
    
def list_cameras(show=False, n=1):
    
    devs = {}
    for dev in enumerate_cameras():
        name = cam_info_to_ref_str(dev)
        if not (name in devs) and not ('PureThermal' in name):
            record = {}
            record['info'] = dev
            record['name'] = name
            devs[name] = record

    return devs

def cam_info_to_ref_str(caminfo):
    ref = caminfo.name + '_' + caminfo.path.split('/')[-1]
    return ref


def list_thorlabs_cameras():
    devs = []
    try:

        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            for cam in available_cameras:
                devs.append(cam)
    except Exception as e:
        print(f'Error listing Thorlabs cameras: {e}')
    return(devs)

def get_ports():
    ports = list_ports.comports()
    out = []
    for port in ports:
        info = {}
        info['id'] = port[0]
        info['description'] = port[1]
        info['details'] = port[2]
        out.append(info)
    return(out)
        
    
    
def connect_tcam_wifi(ip=None):
    if ip is None:
        print('No IP address for WiFi thermal camera specified')
        return (None)
    tcam=TCam()
    tcam.connect(ip)
    return (tcam)

class input_device():

    def __init__(self, parent='NoneAssigned', name=None, device_path = None, deviceID=None, device_type=None, view=None, flipud=False, fliplr = False, save_path=None, interval=0.1):
        self.name=name
        self.deviceID=deviceID
        self.device_type=device_type
        self.device_path = device_path
        self.interval=interval
        self.video_capture=None
        self.view=view
        self.timer=pg.QtCore.QTimer()
        self.timer.setTimerType(0)
        self.timer.timeout.connect(self.capture)
        self.flipud=flipud
        self.fliplr = fliplr
        self.parent = parent
        self.levels= None
        self.median_blur_display = False
        self.show_elapsed_time = True
        self.show_frame_rate = True
        self.show_elapsed_frames = False
        if self.device_type == 'usb_camera':
             pass
        if self.device_type == 'ip_camera':
             self.is_flir = True
        if self.device_type == 'serial':
             self.display_duration = 10
             self.display_max = 10
             self.display_min = -0.5
             self.auto_scale_y = False
             self.display_marker_pos = 2
        if self.device_type == 'thor_camera':
            self.exposure = 65
            self.levels =[200,3000]
            self.xbinning = 2
            self.ybinning = 2
            self.median_blur_display = False 
            self.hot_pixel_correction_enabled = True
            self.hot_pixel_correction_threshold = 700
            
            #self.display_max = 1000
            #self.display_min = 0
           # self.flipud=False
           # self.fliplr=False
        self.counter=0
        self.max_samples=1200 * int(1000/self.interval)  # set max recording to 20 minutes  ## or set at np.inf
        editable_properties = ['interval', 
                               'exposure', 
                               'max_samples', 
                               'display_duration', 
                               'display_max', 
                               'display_min',
                               'auto_scale_y', 
                               'display_marker_pos',
                               'is_flir',
                               'fliplr',
                               'flipud',
                               'xbinning',
                               'ybinning',
                               'median_blur_display',
                               'hot_pixel_correction_enabled',
                               'hot_pixel_correction_threshold',
                               'show_frame_rate',
                               'show_elapsed_time',
                               'show_elapsed_frames']
        self.editable_properties =  []
        for prop in editable_properties:
            if hasattr(self, prop):
                self.editable_properties.append(prop)
                print(f'Property {prop} is editable for device {self.name}')
            else:
                print(f'Property {prop} is not editable for device {self.name}')
       

    def start(self, exp_dir=None, start_time=0, display=True, record=True):
        print(f'Starting device {self.name} with parent {self.parent}')
        self.start_time=start_time
        self.time_points=[]
        self.time_points_non_numeric = []

        self.output_file=os.path.join(exp_dir, (self.name + '.tif'))
        self.time_file=os.path.join(exp_dir, (self.name + '_time.txt'))
        self.display = display
        self.record = record

        if self.device_type == 'serial':
            self.data_stream = []
            self.data_buffer = []
            self.data_stream_non_numeric = []
            self.time_points_non_numeric = []
            self.data_stream_non_numeric.append('start')
            self.time_points_non_numeric.append(time.time()-self.start_time)
            self.output_file = os.path.join(exp_dir, self.name+'.txt')
            self.output_file_non_numeric = os.path.join(exp_dir, self.name+'_nn.txt')
            self.time_file_non_numeric =  os.path.join(exp_dir, self.name+'_nn_time.txt')
            self.serial_connection = serial.Serial(self.device_path, 9600)
            if self.auto_scale_y:
                self.view.enableAutoRange(axis = 'y')
            else:
                self.view.setYRange(self.display_min, self.display_max)
        elif self.record:
            self.writer=imageio.get_writer(self.output_file, mode='I', bigtiff=True)
        else:
            self.writer = None
        if self.device_type == 'usb_camera':
            print(f'Opening usb_camera {self.device_path}')
            self.video_capture=cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

            if self.video_capture.isOpened():
                print(f'Opened camera {self.name}')
            else:
                print(f'Could not open camera {self.name}')
                return
            
        elif self.device_type == 'flir_camera':
            self.video_capture=cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
            self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
            self.video_capture.set(cv2.CAP_PROP_CONVERT_RGB, 0)

        elif self.device_type == 'ip_camera':
            self.video_capture=connect_tcam_wifi(self.deviceID)
            ret=self.video_capture.start_stream()
            print(f'Starting ip cam {self.deviceID} returned {ret}')
            if not ret:
                return
        
        elif self.device_type == 'thor_camera':
           
            self.sdk = self.parent.sdk

            if self.parent.arduino_triggered_thorcam:
                print('Connecting to thorcam arduino...')
                try: 
                    self.arduino = serial.Serial(self.parent.thorcam_arduino_path, baudrate=9600, timeout=1)
                    exposure = int(self.exposure)
                    interpulse = int(self.interval - self.exposure)
                    send_startup_parameters(VAL1 = exposure, VAL2=interpulse)
                except:
                     'Running thorlabs camera in free run mode'

            else:
                'Running thorlabs camera in free run mode'

            print(f'Connecting to ThorLabs Camera {self.deviceID}')
            CameraMap = self.sdk.discover_available_cameras()
            print(type(CameraMap))
            for item in CameraMap:
                if item == self.deviceID:
                    cameraID = item
            
            self.video_capture = self.sdk.open_camera(cameraID)

          ### default settings, no arduino board
            #self.video_capture.frames_per_trigger_zero_for_unlimited = 0 
            #self.video_capture.operation_mode = 1  # 2 for bulb mode

          ### settings to test arduion board
            if self.parent.arduino_triggered_thorcam:
                self.video_capture.frames_per_trigger_zero_for_unlimited = 1 
                self.video_capture.operation_mode = 2  # 2 for bulb mode
            else:
                self.video_capture.frames_per_trigger_zero_for_unlimited = 0 
                self.video_capture.operation_mode =  0 # 2 for bulb mode
                
            print(f'CERNA operation mode: {self.video_capture.operation_mode}')
        
            
            self.video_capture.exposure_time_us = self.exposure *1000  # convert from ms to us
            self.video_capture.hot_pixel_correction_threshold = self.hot_pixel_correction_threshold
            self.video_capture.is_hot_pixel_correction_enabled = self.hot_pixel_correction_enabled

            self.video_capture.binx = self.xbinning
            self.video_capture.biny = self.ybinning
    
            self.video_capture.image_poll_timeout_ms = 0
            self.video_capture.arm(2)
            self.video_capture.issue_software_trigger()


        if self.show_elapsed_time:
            elapsed = 0
            self.timeText = pg.TextItem(f'Time', color = [0,255,0])
            self.view.getViewBox().addItem(self.timeText)
            
            
        if self.show_frame_rate:
            frame_rate = 0
            self.rateText = pg.TextItem(f'Frame rate', color = [255,0,0])
            self.view.getViewBox().addItem(self.rateText)

        if self.show_elapsed_frames:
            frames = 0
            self.frameText = pg.TextItem(f'Frame count', color = [0,0,255])
            self.view.getViewBox().addItem(self.rateText)

       


        self.timer.start(self.interval)


    def capture(self):
        im=None
        print(self.devicetype)
        if self.device_type == 'serial':
            ##TODO
            bytesWaiting = self.serial_connection.read(self.serial_connection.inWaiting())
            self.data_buffer = bytesWaiting.decode()
            self.serial_connection.reset_input_buffer()
           # print(bytesWaiting)
            #print(self.data_buffer)
            lines = self.data_buffer.split('\n')
            print(f'Lines: {lines}')
            #if self.counter > 10:
             #   pdb.set_trace()
            if len(lines)>1:
                reading = lines[-2]
            else:
                reading = 'reset'
            print(f'Reading: {reading}')
            if ' g'in reading:
                reading = reading.split('g')[0]
            timepoint=time.time() - self.start_time
            
            try:
                self.data_stream.append(float(reading))
                self.time_points.append(timepoint)
            except:
                self.data_stream_non_numeric.append(reading)
                self.time_points_non_numeric.append(timepoint)
               # print(reading)
            display_frames = int(self.display_duration*(1000/self.interval))*-1
            #print(display_frames)
            self.view.clear()
            self.view.plot(self.time_points[display_frames:], self.data_stream[display_frames:])
            self.counter = self.counter+1
            return(bytesWaiting)
        elif self.device_type == 'usb_camera':
            ret, frame=self.video_capture.read()
            if ret:
                im=frame[:, :, ::-1]
                im=np.rot90(im, k=3)
            else:
                pass
                #print('no im')
        elif self.device_type == 'flir_camera':
            try:
                ret, frame=self.video_capture.read()
                print(frame.shape)
            except:
                print('Error reading from flir camera')
            if ret:
                im = frame[:-2, :]
            else:
                pass
                #print('no im')
        elif self.device_type == 'ip_camera':
            if self.video_capture.frameQueue.empty():
                time.sleep(0.05)
                #print('ipcam is empty')
                return(im)
            else:
                rawim=self.video_capture.get_image()
                dimg=base64.b64decode(rawim["radiometric"])
                im=im=np.reshape(np.array(array('H', dimg)), shape=[120, 160])
                im=(im/100)-273.15  # convert to deg C
        elif self.device_type == 'thor_camera':
            frame = self.video_capture.get_pending_frame_or_null()
            
            if frame is not None:
                im = frame.image_buffer.transpose()
                
               # print(f'Frame captured from thor camera')
            else:
                print('No frame received from ThorLabs camera')
  
                im = None
  
                
        if im is not None:
            timepoint=time.time() - self.start_time
            self.time_points.append(timepoint)
            #print(f'{self.flipud=}')
            if self.fliplr:
                        im = np.fliplr(im)
            if self.flipud:
                        im = np.flipud(im)
            
            if self.median_blur_display:
                self.view.setImage(cv2.medianBlur(im, 3))
            else:
                self.view.setImage(im)

            if self.show_elapsed_time:
                elapsed = time.time() - self.start_time
                self.timeText.setText(f'T(s): {elapsed}')
                self.timeText.setPos(10,im.shape[1]-20)
            if self.show_frame_rate:
                if len(self.time_points)>1:
                    rate = 1/(self.time_points[-1] - self.time_points[-2])
                else:
                    rate = 0
                self.rateText.setText(f'FPS: {rate}')
                self.rateText.setPos(10, im.shape[1]-40)
            if self.show_elapsed_frames:
                frames = len(self.time_points)
                self.frameText.setText(f'#: {frames}')
                self.frameText.setPos(10,im.shape[1]-60)

        if not (im is None) and self.record:
            self.writer.append_data(im)
            

        self.counter=self.counter + 1
        #print(f'{self.name} count {self.counter}')
        if self.counter > self.max_samples:
            self.stop()


    def stop(self):
        self.timer.stop()
        if self.record:
            np.savetxt(self.time_file, self.time_points, delimiter=',')
        if self.device_type == 'usb_camera' or self.device_type == 'flir_camera':
            self.video_capture.release()
        elif self.device_type == 'ip_camera':
            self.video_capture.stop_stream()
            self.video_capture.shutdown()
        elif self.device_type == 'thor_camera':
            self.video_capture.disarm()
            self.video_capture.roi = (0, 0, 1280, 1024)  # reset roi to full frame
            self.video_capture.dispose()
            #self.video_capture.close()
            #print('Disposing of ThorLabs SDK')
            #self.sdk.dispose()
            if hasattr(self, 'arduino'):
                self.arduino.close()
        elif self.device_type == 'serial':
            pass
            ## TODO
            self.serial_connection.close()
            ## close stream
            ## write data to disk
            self.data_stream_non_numeric.append('end')
            self.time_points_non_numeric.append(time.time()-self.start_time)
            
            if self.record:
                np.savetxt(self.time_file_non_numeric, self.time_points_non_numeric, delimiter=',')         
                np.savetxt(self.time_file, self.time_points, delimiter=',')
                np.savetxt(self.output_file_non_numeric, self.data_stream_non_numeric, delimiter=',', fmt='%s')
                np.savetxt(self.output_file, self.data_stream, delimiter=',')
        if self.show_elapsed_time:
            self.view.getViewBox().removeItem(self.timeText)
        if self.show_frame_rate:
            self.view.getViewBox().removeItem(self.rateText)          
        if self.show_elapsed_frames:
            self.view.getViewBox().removeItem(self.frameText)           
        print('Acquisition completed')


def decode_telemetry(image, mode="footer"):
        """
        Extracts telemetry from an image
        """
        res = struct.unpack("<2cII16x4h6xIh2xh8xhI4xhhhhhh64xI172x", image[-2, :])

        telemetry = res

        major_version = res[0]
        minor_version = res[1]
        uptime_ms = res[2]
        status = res[3]
        revision = res[4:8]
        frame_count = res[8]
        frame_mean = res[9]
        fpa_temp_k = res[10] / 100.0
        ffc_temp_k = res[11] / 100.0
        ffc_elapsed_ms = res[12]
        agc_roi = res[13:17]
        agc_clip_high = res[17]
        agc_clip_low = res[18]
        vudeo_format = res[19]

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
        start = time.time()
        while True:
            line = ser.readline().decode('utf-8').strip()
            if "READY" in line:
                print("Arduino is ready.")
                break
            if time.time()-start > 8:
                print('Could not connect')
                return
            
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