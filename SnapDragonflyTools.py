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

def combine_all_movies(rd=None, filt = '**/174*', resolution=0.1, fps=30, figsize=(6, 4), bbox=[43, 18, 80, 80], flir_range=[5, 55], time_range=None, plot_hist=False, plot_temp_curve=True):
    if rd is None:
        return()
        #rd = 'M:/Biology2/Team folder - Mark/David - Primary data/Behavior/HC Plate/20250220'
    ex_dirs = glob.glob(filt, root_dir=rd, recursive=True)
    At=None
    if plot_temp_curve:
        Ft = plt.figure(figsize =(5,5))
        At = Ft.add_subplot(1,1,1)
    for x in ex_dirs:
        ex = os.path.join(rd,x)
        make_combined_movie(input_dir=ex, resolution = resolution, fps = fps, time_range=time_range, plot_hist=plot_hist, At=At,flir_range=flir_range, plot_temp_curve= plot_temp_curve)
      
    
def make_combined_movie(input_dir=None, resolution=0.1, fps=30, figsize=(6, 4), bbox=[43, 18, 80, 80], plot_hist=True, flir_range=[5,55], time_range=None, mask_std = 4, convert_K_to_C=False, plot_temp_curve=True, At=None):
    F = plt.figure(figsize=figsize)
   
    #rect = patches.Rectangle([bbox[0], bbox[1]], bbox[2],
     #                        bbox[3], linewidth=1, edgecolor='w', facecolor='none')
    if plot_temp_curve:
        if At is None:
            Ft = plt.figure(figsize =(5,5))
            At = Ft.add_subplot(1,1,1)
    if plot_hist:
        A_hist = F.add_axes([0, 0, 0.9, 0.3])
        A_beh = F.add_axes([0, 0.3, 0.45, 0.7])
        A_flir = F.add_axes([0.45, 0.3, 0.45, 0.7])
        A_scale = F.add_axes([0.91, 0.3975, 0.02, 0.5075])
    else:
       
        A_beh = F.add_axes([0, 0, 0.45, 1])
        A_flir = F.add_axes([0.45, 0, 0.45, 1])
        A_scale = F.add_axes([0.91, 0.0975, 0.02, 0.8075])
    A_flir.set_axis_off()
   
    scale = np.zeros([256, 8])
    for h in range(8):
        scale[:, h] = np.linspace(255, 0, num=256)
    A_scale.imshow(scale, cmap='plasma')
    A_scale.set_aspect('auto')
    A_scale.set_axis_off()
    A_scale.text(10,0,f'{flir_range[1]} °C', verticalalignment = 'top')
    A_scale.text(10,255,f'{flir_range[0]} °C', verticalalignment = 'bottom')
   

    if input_dir is None:
        input_dir = 'M:/Biology2/Team folder - Mark/David - Primary data/Behavior/HC Plate/20250220/Post-zymosan/Mouse 5/1740084751'
        input_dir = 'C:/Users/david.yarmolinsky/OneDrive - Dragonfly Therapeutics/Hot cold plate cipn data/20250626/Cold plate Cisplatin/58/1750960637'
        
    output_file = os.path.join(input_dir, 'Combined movie.mp4')
    output_data_file = os.path.join(input_dir, 'Data.csv')
    temp_center = np.mean(flir_range)
    try:
        input_behavior_file = os.path.join(
            input_dir, glob.glob('*Ard*.tif', root_dir=input_dir)[0])
        input_flir_file = os.path.join(
            input_dir, glob.glob('*10.*.tif', root_dir=input_dir)[0])
        input_behavior_time = os.path.join(
            input_dir, glob.glob('*Ard*.txt', root_dir=input_dir)[0])
        input_flir_time = os.path.join(
            input_dir, glob.glob('*10.*.txt', root_dir=input_dir)[0])
    except:
        print(f'Data not valid for folder {input_dir}, skipping')
        return([],[])
    beh_time = np.genfromtxt(input_behavior_time, delimiter=',')
    flir_time = np.genfromtxt(input_flir_time, delimiter=',')
    if np.size(flir_time) <2 or np.size(beh_time) <2:
        print(f'Data not valid for folder {input_dir} ')
        return([],[])
    all_time = np.concatenate([beh_time, flir_time])
    # try:
    #     all_time = np.concatenate([beh_time, flir_time])
    # except:
    #     pdb.set_trace()
    if time_range is None:
        start = np.min(all_time)
        stop = np.amax(all_time)
    else:
        start = time_range[0]
        stop = time_range[1]
    duration = stop-start
    time_scale = np.linspace(start, stop, num=int(duration/resolution))
    beh_reader = imageio.get_reader(input_behavior_file)
    flir_reader = imageio.get_reader(input_flir_file)
    
    flir_mask = np.std(np.array(imageio.mimread(input_flir_file)), axis=0) > mask_std

    writer=imageio.get_writer(output_file, fps=fps)
    output_matrix = np.zeros([time_scale.shape[0], 2])
    output_matrix[:,0] = time_scale
    for tt, t in enumerate(time_scale):
        ex_time = t-start
        f_index = np.searchsorted(flir_time, t, side='left')
        if f_index >= flir_time.shape[0]:
            f_index = f_index-1
        f_time = flir_time[f_index]
        f_frame = flir_reader.get_data(f_index)
     #   left = bbox[0]
     #   right = bbox[0] + bbox[2]
     #   top = bbox[1]
     #   bot = bbox[1] + bbox[3]
     #   spot = f_frame[top:bot, left:right]
        spot = f_frame[flir_mask]
        spot = spot.flatten()
        # if convert_K_to_C:
        #    spot = spot/100
        #    spot = spot - 273.3
        if convert_K_to_C:
            f_frame = convert_flir(f_frame, flir_range=flir_range)
        b_index = np.searchsorted(beh_time, t, side='left')
        if b_index >= beh_time.shape[0]:
            b_index = b_index-1
        b_time = beh_time[f_index]
        b_frame = beh_reader.get_data(b_index)
        b_frame = np.rot90(b_frame, k=1)

        A_beh.clear()
        A_beh.imshow(b_frame)
        A_beh.text(5,5, f'Time (s): {np.round(ex_time, decimals=1)}', verticalalignment = 'top', color = [0,0.5,1])
        A_beh.set_axis_off()

        A_flir.clear()
        A_flir.imshow(f_frame*flir_mask, cmap = 'plasma', vmin = flir_range[0], vmax = flir_range[1])
        A_flir.set_axis_off()
        #return(f_frame)
        #A_flir.add_patch(rect)
        
        
        # hist, edges = np.histogram(
        #     spot, range=flir_range, bins=100, density=True)
        # max_bin = np.argmax(hist)
        # temp1 = edges[max_bin]
        # temp2 = edges[max_bin+1]
        # temp = np.mean([temp1, temp2])
        # medtemp = np.median(spot)
        modetemp = mode(np.round(spot, decimals=1))[0]
        output_matrix[tt,1] = modetemp
        A_beh.text(5,20, f'Temperature: {np.round(modetemp, decimals=1)} °C',  verticalalignment = 'top',color='r')
        if plot_hist:
            A_hist.clear()
           
            A_hist.hist(spot, range=flir_range, color='k', bins=100, density=True)
           
            A_hist.text(temp_center, 2, f'Temperature: {np.round(modetemp, decimals=1)} °C', horizontalalignment = 'center', verticalalignment = 'top')
            A_hist.text(temp_center, 1.8, f'Mode temperature: {np.round(modetemp, decimals=1)} °C', horizontalalignment = 'center', verticalalignment = 'top')
         #   A_hist.plot([temp, temp], [0, 2], color='r')
            A_hist.plot([modetemp, modetemp], [0, 2], color='r')


        capture = figure_to_np(F)
        writer.append_data(capture)
        print(f'Writing frame {tt} of {time_scale.shape[0]}')
    
    if plot_temp_curve:
        At.plot(output_matrix[:,0],output_matrix[:,1])
    plt.close(F)
    writer.close()
    np.savetxt(output_data_file, output_matrix, delimiter=',')
    beh_reader.close()
    flir_reader.close()
    print(output_file)
    return (beh_time, flir_time)




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

