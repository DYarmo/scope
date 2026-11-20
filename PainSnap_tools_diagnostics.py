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
        print(name)
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
        info['name'] = info['id']
        print({port.serial_number})
        #print(port[3])
        #print(port[4])
        #print(port[5])
        if port.serial_number in port_aliases():
            info['name'] = port_aliases()[port.serial_number]
        info['serial_number'] = port.serial_number
        out.append(info)
    return(out)
        
def port_aliases():
    aliases = {}
    aliases['8503631353035111A2C0'] = 'Force sensor'
    aliases['03535343335351903271'] = 'Arduino control'
    return(aliases)


def connect_tcam_wifi(ip=None):
    if ip is None:
        print('No IP address for WiFi thermal camera specified')
        return (None)
    tcam=TCam()
    tcam.connect(ip)
    return (tcam)


class input_device():

    def __init__(self, parent='NoneAssigned', name=None, device_path = None, deviceID=None, device_type=None, view=None, flipud=False, fliplr = False, save_path=None, interval=0.1, calibration_factor=1.0):
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
        self.save_format = 'tif'
        self.calibration_factor = calibration_factor
        if self.device_type == 'usb_camera':
             pass
             self.save_format = 'avi'
        if self.device_type == 'ip_camera':
             self.is_flir = True
        if self.device_type == 'serial':
             self.display_duration = 10
             self.display_max = 10
             self.display_min = -0.5
             self.auto_scale_y = False
             self.display_marker_pos = 2
             self.save_format = 'txt'
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

        # Acquisition diagnostics
        # These are intentionally lightweight and summarize performance every
        # ~5 seconds rather than printing on every frame.
        self.diag_enabled = True
        self.diag_summary_interval = 5.0
        self.diag_last_summary = None
        self.diag_callbacks = 0
        self.diag_successful_frames = 0
        self.diag_failed_frames = 0
        self.diag_callback_intervals = []
        self.diag_acquire_times = []
        self.diag_process_times = []
        self.diag_display_times = []
        self.diag_write_times = []
        self.diag_total_times = []
        self.diag_last_callback_time = None
        self.diag_grab_times = []
        self.diag_retrieve_times = []

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
                               'show_elapsed_frames',
                               'save_format',
                               'calibration_factor'
                               ]
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

        # Reset acquisition diagnostics for this run
        self.diag_last_summary = time.perf_counter()
        self.diag_callbacks = 0
        self.diag_successful_frames = 0
        self.diag_failed_frames = 0
        self.diag_callback_intervals = []
        self.diag_acquire_times = []
        self.diag_process_times = []
        self.diag_display_times = []
        self.diag_write_times = []
        self.diag_total_times = []
        self.diag_last_callback_time = None
        self.diag_grab_times = []
        self.diag_retrieve_times = []

        self.output_file=os.path.join(exp_dir, (self.name + '.' + self.save_format))
        self.time_file=os.path.join(exp_dir, (self.name + '_time.txt'))
        self.display = display
        self.record = record

        ### Start serial device - open connection and create output files
        if self.device_type == 'serial':
            self.data_stream = []
            self.data_buffer = ''
            self.data_stream_non_numeric = []
            self.time_points_non_numeric = []
            self.data_stream_non_numeric.append('start')
            self.time_points_non_numeric.append(time.time()-self.start_time)
            self.output_file = os.path.join(exp_dir, self.name+'.' + self.save_format)
            self.output_file_non_numeric = os.path.join(exp_dir, self.name+'_nn.txt')
            self.time_file_non_numeric =  os.path.join(exp_dir, self.name+'_nn_time.txt')
            try:
                self.serial_connection = serial.Serial(self.device_path, 9600)
                print(f'Opened serial connection to {self.device_path}')
            except serial.SerialException as e:
                print(f'Error opening serial connection to {self.device_path}: {e}')
                self.stop()
            if self.auto_scale_y or 1==1:
                self.view.enableAutoRange(axis = 'y')
            else:
                self.view.setYRange(self.display_min, self.display_max)
        elif self.record:
            if self.save_format == 'avi':
                fps = int(1000/self.interval)
                #self.writer = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'MJPG'), fps, (800, 600))
                self.writer = imageio.get_writer(self.output_file, fps=fps)
            elif self.save_format == 'tif':
                self.writer=imageio.get_writer(self.output_file, mode='I', bigtiff=True)
        else:
            self.writer = None
        if self.device_type == 'usb_camera':
            print(f'Opening usb_camera {self.device_path}')
            self.video_capture=cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)
            self.video_capture.set(cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'MJPG'))
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)

            print("Requested FPS 30")
            print("Actual FPS:", self.video_capture.get(cv2.CAP_PROP_FPS))

            print("\n=== USB CAMERA DIAGNOSTICS ===")
            print("Backend:", self.video_capture.getBackendName())
            print("Width:", self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            print("Height:", self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("FPS:", self.video_capture.get(cv2.CAP_PROP_FPS))
            print("FOURCC:", self.video_capture.get(cv2.CAP_PROP_FOURCC))
            print("Buffer size:", self.video_capture.get(cv2.CAP_PROP_BUFFERSIZE))
            fourcc = int(self.video_capture.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = ''.join([
            chr(fourcc & 0xFF),
            chr((fourcc >> 8) & 0xFF),
            chr((fourcc >> 16) & 0xFF),
            chr((fourcc >> 24) & 0xFF)
            ])
            print("FOURCC decoded:", fourcc_str)

            print("==============================\n")



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
                    self.arduino = serial.Serial(self.parent.thorcam_arduino_path, baudrate=115200, timeout=1)
                    exposure = int(self.exposure)
                    interpulse = int(self.interval - self.exposure)
                    send_startup_parameters(VAL1 = exposure, VAL2=interpulse, port=self.parent.thorcam_arduino_path, baudrate=115200)
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
        """Acquire one sample/frame and collect lightweight performance diagnostics."""
        total_t0 = time.perf_counter()
        im = None

        # Timer callback diagnostics
        callback_t0 = total_t0
        self.diag_callbacks += 1
        if self.diag_last_callback_time is not None:
            self.diag_callback_intervals.append(
                callback_t0 - self.diag_last_callback_time
            )
        self.diag_last_callback_time = callback_t0

        # ------------------------------------------------------------
        # Acquisition
        # ------------------------------------------------------------
        acquire_t0 = time.perf_counter()

        if self.device_type == 'serial':
            # Process complete lines; retain any incomplete line for the next call.
            reading = None
            bytes_waiting = self.serial_connection.read(self.serial_connection.in_waiting)
            self.data_buffer += bytes_waiting.decode()
            while '\n' in self.data_buffer:
                line, self.data_buffer = self.data_buffer.split('\n', 1)
                line = line.strip()  # Removes \r and whitespace

                if not line:
                    continue

                try:
                    reading = float(line)
                    reading = reading * self.calibration_factor
                except ValueError:
                    continue

                # Handle each complete numeric reading here.
                print(f"Reading: {reading}")

            #print(reading)
            timepoint = time.time() - self.start_time

            try:
                self.data_stream.append(float(reading))
                self.time_points.append(timepoint)
            except:
                self.data_stream_non_numeric.append(reading)
                self.time_points_non_numeric.append(timepoint)

            acquire_dt = time.perf_counter() - acquire_t0
            self.diag_acquire_times.append(acquire_dt)

            display_t0 = time.perf_counter()
            display_frames = int(self.display_duration * (1000/self.interval)) * -1
            self.view.clear()
            self.view.plot(
                self.time_points[display_frames:],
                self.data_stream[display_frames:]
            )
            display_dt = time.perf_counter() - display_t0
            self.diag_display_times.append(display_dt)

            self.counter += 1
            self.diag_successful_frames += 1

            total_dt = time.perf_counter() - total_t0
            self.diag_total_times.append(total_dt)
            self._print_diagnostics_if_due()


        elif self.device_type == 'usb_camera':
            grab_t0 = time.perf_counter()
            grab_ok = self.video_capture.grab()
            grab_dt = time.perf_counter() - grab_t0

            retrieve_t0 = time.perf_counter()
            ret, frame = self.video_capture.retrieve()
            retrieve_dt = time.perf_counter() - retrieve_t0

            if ret:
                im = frame[:, :, ::-1]
                im = np.rot90(im, k=3)
            self.diag_grab_times.append(grab_dt)
            self.diag_retrieve_times.append(retrieve_dt)

        elif self.device_type == 'flir_camera':
            try:
                ret, frame = self.video_capture.read()
            except Exception:
                print('Error reading from flir camera')
                ret = False
                frame = None

            if ret:
                im = frame[:-2, :]

        elif self.device_type == 'ip_camera':
            if self.video_capture.frameQueue.empty():
                # NOTE: This sleep remains unchanged for diagnostic purposes.
                # It will be a candidate for removal in the next optimization step.
                time.sleep(0.05)
                acquire_dt = time.perf_counter() - acquire_t0
                self.diag_acquire_times.append(acquire_dt)
                self.diag_failed_frames += 1
                self.diag_total_times.append(time.perf_counter() - total_t0)
                self._print_diagnostics_if_due()
                return im
            else:
                rawim = self.video_capture.get_image()
                dimg = base64.b64decode(rawim["radiometric"])
                im = np.reshape(
                    np.array(array('H', dimg)),
                    shape=[120, 160]
                )
                im = (im/100) - 273.15

        elif self.device_type == 'thor_camera':
            frame = self.video_capture.get_pending_frame_or_null()

            if frame is not None:
                im = frame.image_buffer.transpose()
            else:
                im = None

        acquire_dt = time.perf_counter() - acquire_t0
        self.diag_acquire_times.append(acquire_dt)

        if im is None:
            self.diag_failed_frames += 1
            self.counter += 1

            total_dt = time.perf_counter() - total_t0
            self.diag_total_times.append(total_dt)
            self._print_diagnostics_if_due()
            return

        self.diag_successful_frames += 1

        # ------------------------------------------------------------
        # Processing
        # ------------------------------------------------------------
        process_t0 = time.perf_counter()

        timepoint = time.time() - self.start_time
        self.time_points.append(timepoint)

        if self.fliplr:
            im = np.fliplr(im)
        if self.flipud:
            im = np.flipud(im)

        if self.median_blur_display:
            display_image = cv2.medianBlur(im, 3)
        else:
            display_image = im

        process_dt = time.perf_counter() - process_t0
        self.diag_process_times.append(process_dt)

        # ------------------------------------------------------------
        # Display
        # ------------------------------------------------------------
        display_t0 = time.perf_counter()

        self.view.setImage(display_image)

        if self.show_elapsed_time:
            elapsed = time.time() - self.start_time
            self.timeText.setText(f'T(s): {elapsed}')
            self.timeText.setPos(10, im.shape[1]-20)

        if self.show_frame_rate:
            if len(self.time_points) > 1:
                dt = self.time_points[-1] - self.time_points[-2]
                rate = 1/dt if dt > 0 else 0
            else:
                rate = 0
            self.rateText.setText(f'FPS: {rate}')
            self.rateText.setPos(10, im.shape[1]-40)

        if self.show_elapsed_frames:
            frames = len(self.time_points)
            self.frameText.setText(f'#: {frames}')
            self.frameText.setPos(10, im.shape[1]-60)

        display_dt = time.perf_counter() - display_t0
        self.diag_display_times.append(display_dt)

        # ------------------------------------------------------------
        # Disk write
        # ------------------------------------------------------------
        write_t0 = time.perf_counter()

        if self.record:
            self.writer.append_data(im)

        write_dt = time.perf_counter() - write_t0
        self.diag_write_times.append(write_dt)

        # ------------------------------------------------------------
        # Finish + diagnostics
        # ------------------------------------------------------------
        self.counter += 1

        if self.counter > self.max_samples:
            self.stop()

        total_dt = time.perf_counter() - total_t0
        self.diag_total_times.append(total_dt)

        self._print_diagnostics_if_due()

    def _print_diagnostics_if_due(self):
        """Print a compact acquisition-performance summary every ~5 seconds."""
        if not self.diag_enabled:
            return

        now = time.perf_counter()
        if self.diag_last_summary is None:
            self.diag_last_summary = now
            return

        if now - self.diag_last_summary < self.diag_summary_interval:
            return

        def stats(values):
            if not values:
                return "n/a"
            avg = 1000 * float(np.mean(values))
            mx = 1000 * float(np.max(values))
            return f"{avg:.2f} ms avg / {mx:.2f} ms max"

        def rate_from_intervals(values):
            if not values:
                return 0.0
            mean_dt = float(np.mean(values))
            return 1.0 / mean_dt if mean_dt > 0 else 0.0

        elapsed = now - self.diag_last_summary
        callback_rate = self.diag_callbacks / elapsed

        print("\n" + "=" * 70)
        print(f"PainSnap diagnostics: {self.name} [{self.device_type}]")
        print(f"Requested interval: {self.interval:.1f} ms "
              f"({1000/self.interval:.2f} Hz)")
        print(f"Callbacks: {self.diag_callbacks} "
              f"({callback_rate:.2f} callbacks/sec)")
        print(f"Successful frames: {self.diag_successful_frames}")
        print(f"Failed/empty frames: {self.diag_failed_frames}")
        print(f"Observed callback interval: "
              f"{1000 / rate_from_intervals(self.diag_callback_intervals):.2f} ms"
              if self.diag_callback_intervals else
              "Observed callback interval: n/a")
        print(f"Acquire:  {stats(self.diag_acquire_times)}")
        print(f"Process:  {stats(self.diag_process_times)}")
        print(f"Display:  {stats(self.diag_display_times)}")
        print(f"Write:    {stats(self.diag_write_times)}")
        print(f"Grab:     {stats(self.diag_grab_times)}")
        print(f"Retrieve: {stats(self.diag_retrieve_times)}")
        print(f"Total:    {stats(self.diag_total_times)}")
        print("=" * 70)
        
        # Start a fresh measurement window.
        self.diag_last_summary = now
        self.diag_callbacks = 0
        self.diag_successful_frames = 0
        self.diag_failed_frames = 0
        self.diag_callback_intervals = []
        self.diag_acquire_times = []
        self.diag_process_times = []
        self.diag_display_times = []
        self.diag_write_times = []
        self.diag_total_times = []
        self.diag_grab_times = []
        self.diag_retrieve_times = []

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

        First line unpacks 320 bytes of embedded binary metadata from the second-to-last row of an image array (image[-2, :]) 
        into a tuple of 18 distinct values.

        The format string "<2cII16x4h6xIh2xh8xhI4xhhhhhh64xI172x" uses little-endian byte order (<) and expects an input buffer of exact size 320 bytes (e.g., a row of 320 uint8 pixels, or 160 uint16 pixels):

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
        print("No port specified for Arduino communication. Please provide a valid port.")
        return False
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