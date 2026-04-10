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


def convert_tiff_to_avi(input_file=None, extension='.avi', add_scale=False, convert_FLIR=False, fps=30, rotate=False, flir_range=[25, 45]):
    if not ('.tif' in input_file):
        print('Input file must have extension ".tif"')
        return

    output_file=input_file.replace('.tif', extension)
    reader=imageio.get_reader(input_file)
    first_frame=reader.get_data(0)
    h=first_frame.shape[0]
    w=first_frame.shape[1]
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    writer=imageio.get_writer(output_file, fps=30)
    fps=30
    # writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))
    F=plt.figure()
    Aim=F.add_axes([0, 0, 1, 1])
    for n in range(reader.get_length()):
        frame=reader.get_data(n)
        if rotate:
            frame=np.rot90(frame, k=1)
        if convert_FLIR:
            frame=frame/100
            frame=frame - 273.15
            frame=frame-flir_range[0]
            frame=frame/(flir_range[1]-flir_range[0])
            frame[frame > 1]=1
            frame[frame < 0]=0
            frame=frame * 255
            frame=frame.astype(np.uint8)
            # pdb.set_trace()
        if add_scale:
            scale=np.zeros([h, 8])
            for i in range(8):
                scale[:, i]=np.linspace(255, 0, num=h)
            frame=np.hstack((frame, scale))
        Aim.clear()
        Aim.set_axis_off()
        Aim.imshow(frame)
       # pdb.set_trace()
        capture=figure_to_np(F)
        writer.append_data(capture)
        # writer.write(frame)
    reader.close()
    writer.close()
    # writer.release()

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

def list_cameras(show=False, n=1):
    
    start=time.time()
    #devs=FilterGraph().get_input_devices()
    devs = enumerate_cameras()
    
    if show:
        F=plt.figure()
        for i, dev in enumerate(devs):
            cap=cv2.VideoCapture(i)


            if cap.isOpened():
               print(f'Capture opened for {dev}')
            else:
               print(f'Unable to capture from {dev}')
            for j in range(n):
                ret, frame=cap.read()
                if ret:
                    A=F.add_subplot(len(devs), n, (i*n) + j+1)
                    A.imshow(frame)
                    A.set_title(dev)
            cap.release()
    print(time.time()-start)
    return devs


def list_thorlabs_cameras():
    devs = []
    try:

        with TLCameraSDK() as sdk:
            available_cameras = sdk.discover_available_cameras()
            for cam in available_cameras:
                devs.append(cam)
    except Exception as e:
        print(f'Error listing Thorlabs cameras: {e}')
    return devs

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

    def __init__(self, name=None, deviceID=None, device_type=None, view=None, save_path=None, interval=0.1):
        self.name=name
        self.deviceID=deviceID
        self.device_type=device_type
        self.interval=interval
        self.video_capture=None
        self.view=view
        self.timer=pg.QtCore.QTimer()
        self.timer.setTimerType(0)
        self.timer.timeout.connect(self.capture)
         
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
            self.exposure = 99

        self.counter=0
        self.max_samples=1200 * int(1000/self.interval)  # set max recording to 20 minutes  ## or set at np.inf
        editable_properties = ['interval', 'exposure', 'max_samples', 'display_duration', 'display_max', 'display_min', 'auto_scale_y', 'display_marker_pos','is_flir']
        self.editable_properties =  []
        for prop in editable_properties:
            if hasattr(self, prop):
                self.editable_properties.append(prop)
       
       

    def start(self, exp_dir=None, start_time=0):
        self.start_time=start_time
        self.time_points=[]
        self.time_points_non_numeric = []
        self.output_file=os.path.join(exp_dir, self.name+'.tif')
        self.time_file=os.path.join(exp_dir, self.name+'_time.txt')
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
            self.serial_connection = serial.Serial(self.name, 9600)
            if self.auto_scale_y:
                self.view.enableAutoRange(axis = 'y')
            else:
                self.view.setYRange(self.display_min, self.display_max)
        else:
            self.writer=imageio.get_writer(self.output_file, mode='I', bigtiff=True)
        if self.device_type == 'usb_camera':
            print(f'Opening usb_camera {self.deviceID}')
            self.video_capture=cv2.VideoCapture(self.deviceID)

            if self.video_capture.isOpened():
                print(f'Opened camera {self.deviceID}')
            else:
                print(f'Could not open camera {self.deviceID}')
                return
        elif self.device_type == 'ip_camera':
            self.video_capture=connect_tcam_wifi(self.deviceID)
            ret=self.video_capture.start_stream()
            print(f'Starting ip cam {self.deviceID} returned {ret}')
            if not ret:
                return
        elif self.device_type == 'thor_camera':
            self.sdk = TLCameraSDK()
            print(f'Connecting to ThorLabs Camera {self.deviceID}')
            CameraMap = self.sdk.discover_available_cameras()
            print(type(CameraMap))
            for item in CameraMap:
                print(item)
                print(type(item))
                print(f'Checking if camera {item} matches deviceID {self.deviceID}')
                print(f'{item==self.deviceID}')
                print(f'{type(item)} vs {type(self.deviceID)}')
                if item == self.deviceID:
                    cameraID = item
            
            self.video_capture = self.sdk.open_camera(cameraID)
            #self.video_capture = self.sdk.open_camera(CameraMap[0])

            #self.video_capture = self.sdk.open_camera(self.sdk.discover_available_cameras()[self.deviceID])
          
            self.video_capture.frames_per_trigger_zero_for_unlimited = 0 
            #self.video_capture.operation_mode = 1  # 2 for bulb mode
            print(f'CERNA operation mode: {self.video_capture.operation_mode}')
        
            
            self.video_capture.exposure_time_us = self.exposure *1000  # convert from ms to us
            #self.video_capture.hot_pixel_correction_threshold = 655
            #self.video_capture.is_hot_pixel_correction_enabled = True
            try:
                self.video_capture.binx = self.xbinning
            except:
                print("Error occurred while setting xbinning")
            try:
                self.video_capture.biny = self.ybinning
            except:
                print("Error occurred while setting ybinning")  
            self.video_capture.image_poll_timeout_ms = 0
            self.video_capture.arm(2)
            self.video_capture.issue_software_trigger()



        self.timer.start(self.interval)


    def capture(self, display=True):
        im=None
        if self.device_type == 'serial':
            ##TODO
            bytesWaiting = self.serial_connection.read(self.serial_connection.inWaiting())
            self.data_buffer = bytesWaiting.decode()
            self.serial_connection.reset_input_buffer()
            #print(bytesWaiting)
            #print(self.data_buffer)
            lines = self.data_buffer.split('\n')
            print(lines)
            #if self.counter > 10:
             #   pdb.set_trace()
            if len(lines)>1:
                reading = lines[-2]
            else:
                reading = 'reset'
                
            if ' g'in reading:
                reading = reading.split('g')[0]
            timepoint=time.time() - self.start_time
            
            try:
                self.data_stream.append(float(reading))
                self.time_points.append(timepoint)
            except:
                self.data_stream_non_numeric.append(reading)
                self.time_points_non_numeric.append(timepoint)
                print(reading)
            display_frames = int(self.display_duration*(1000/self.interval))*-1
            print(display_frames)
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
                print('no im')
        elif self.device_type == 'ip_camera':
            if self.video_capture.frameQueue.empty():
                time.sleep(0.05)
                print('ipcam is empty')
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
                print(f'Frame captured from thor camera')
            else:
                print('No frame received from ThorLabs camera')
                im = None
        try:
            self.view.setImage(im)
        except:
            print(im)
            print(im.shape)
            print(np.amax(im))
            print(np.min(im))
        if not (im is None):
            self.writer.append_data(im)
            timepoint=time.time() - self.start_time
            self.time_points.append(timepoint)
        self.counter=self.counter + 1
        print(f'{self.name} count {self.counter}')
        if self.counter > self.max_samples:
            self.stop()


    def stop(self):
        self.timer.stop()
        np.savetxt(self.time_file, self.time_points, delimiter=',')
        if self.device_type == 'usb_camera':
            self.video_capture.release()
        elif self.device_type == 'ip_camera':
            self.video_capture.stop_stream()
            self.video_capture.shutdown()
        elif self.device_type == 'thor_camera':
            self.video_capture.disarm()
            self.video_capture.roi = (0, 0, 1280, 1024)  # reset roi to full frame
            self.video_capture.dispose()
            #self.video_capture.close()
            print('Disposing of ThorLabs SDK')
            self.sdk.dispose()
            
        elif self.device_type == 'serial':
            pass
            ## TODO
            self.serial_connection.close()
            ## close stream
            ## write data to disk
            self.data_stream_non_numeric.append('end')
            self.time_points_non_numeric.append(time.time()-self.start_time)
            
            np.savetxt(self.time_file_non_numeric, self.time_points_non_numeric, delimiter=',')         
            np.savetxt(self.time_file, self.time_points, delimiter=',')
            np.savetxt(self.output_file_non_numeric, self.data_stream_non_numeric, delimiter=',', fmt='%s')
            np.savetxt(self.output_file, self.data_stream, delimiter=',')
        print('Acquisition completed')

