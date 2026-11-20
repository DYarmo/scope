# -*- coding: utf-8 -*-
"""
Created on Thurs April 20 2026

@author: david.yarmolinsky
"""
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction, QFileDialog, QLineEdit, QListWidget, QGridLayout, QVBoxLayout, QWidget, QInputDialog
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtCore import Qt, QRectF, pyqtSlot
from PyQt5.QtGui import QPainter
import sys
import cv2
import pyqtgraph as pg
from matplotlib import pyplot as plt
import numpy as np
#import PainSnap_tools as tools
import PainSnap_tools_diagnostics as tools
import time
import os
import serial
from serial.tools import list_ports
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE
import faulthandler
faulthandler.enable()
from rclone_python import rclone

class AcqGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Window setup
        self.setGeometry(10, 60, 1600, 900)  # Adjusted window size
        self.setWindowTitle('PainSnap')
       
        # Main widget and layout
   
        self.GL = pg.GraphicsLayoutWidget()
        self.GL.resize(1600, 700)
        self.GL.move(0, 50)
        self.setCentralWidget(self.GL)
       
        self.ip_list = ["10.50.40.109"]
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.config_menu = self.menu_bar.addMenu('Configure')
        self.acq_menu = self.menu_bar.addMenu('Acquire')

        self.cam_menu = QMenu('Select cameras')
        self.thorcam_menu = QMenu('Select Thorlabs cameras')
        self.ipcam_menu = QMenu('Select ip cameras')
        self.flircam_menu = QMenu('Select FLIR cameras')
        self.port_menu = QMenu('Select serial devices')
        self.property_menu = QMenu('Set device properties...')

        self.start_action = QAction('Acquire', self)
        self.preview_action = QAction ('Preview', self)
        self.stop_action = QAction('End acquisition', self)
        self.stop_action.setEnabled(False)
        self.find_action = QAction('Find inputs', self)
        self.add_ip_action = QAction('Add ip camera...', self)
        self.remove_ip_action = QAction('Remove ip camera', self)

        self.find_action.triggered.connect(self.update_sources)
        self.start_action.triggered.connect(self.start_acq)
        self.preview_action.triggered.connect(self.start_preview)
        self.stop_action.triggered.connect(self.stop_acq)

        self.set_path_action = QAction('Set save path...', self)
        self.set_path_action.triggered.connect(self.getDirectory)
        self.set_remote_path_action = QAction('Set remote path...', self)
        self.set_remote_path_action.triggered.connect(self.getRemoteDirectory)
        self.backup_action = QAction('Backup data to remote repository')
        self.backup_action.triggered.connect(self.backup_data)
        
        self.config_menu.addAction(self.set_path_action)
        self.config_menu.addAction(self.set_remote_path_action)
        self.config_menu.addAction(self.backup_action)

        self.config_menu.addAction(self.find_action)
        self.config_menu.addMenu(self.cam_menu)
        self.config_menu.addMenu(self.thorcam_menu)
        self.config_menu.addMenu(self.ipcam_menu)
        self.config_menu.addMenu(self.flircam_menu)
        self.config_menu.addMenu(self.port_menu)
        self.config_menu.addMenu(self.property_menu)
        self.update_sources()

        self.acq_menu.addAction(self.start_action)
        self.acq_menu.addAction(self.preview_action)
        self.acq_menu.addAction(self.stop_action)

        self.saveDirectory = os.path.normpath(r'/home/rlab-scope/Data/PainSnap data')
        self.remote_repository  = '/run/user/1000/gvfs/sftp:host=painseq,user=david/rlab/data/david/PainSnap Data'
        self.local_repository = os.path.normpath(r'/home/rlab-scope/Data/PainSnap data')
        self.saveText = QLineEdit("Save to: " + self.saveDirectory, self)
        self.saveText.setReadOnly(True)
        self.saveText.resize(1600, 50)
        self.saveText.move(0, 850)
        self.saveText.setStyleSheet("""QLineEdit { background-color
                                    : transparent ; color: blue}""")
        self.active_cams = {}
        self.active_ipcams = {}
        self.active_flircams = {}
        self.active_ports = {}
        self.active_thorcams = {}   
        self.running_thorcams = []
        self.cam_objects = {}
        self.cam_views = {}

        self.thorcam_trial_list = {}
        self.trials_in_review = {}
        
        self.sdk = TLCameraSDK()
        self.show()

    def property_setter(self):
        """
        Generic interface for users to set properties of camera objects via QInputDialog
        To do: option to set properties from list of pre-sets and to constrain input ranges for specific properties
        """
        print('Property setter called')
        
        sender = self.sender().text()
        dev = sender.split('--')[0]
        prop = sender.split('--')[1]
        obj = self.cam_objects[dev]
        print(dev)
        print(prop)
        currentVal = str(getattr(obj, prop))
        d, ok = QInputDialog.getText(self, 'Device properties', f'Set {prop} for {dev}:', text=currentVal)
        try:
            value = eval(d)
        except:
            value = d
        setattr(self.cam_objects[dev], prop, value)


    def closeEvent(self,event):
        for thorcam in self.running_thorcams:
            print(f'Disconnecting from ThorLabs Camera {thorcam}')
            thorcam.dispose()
        self.sdk.dispose()
        print('Goodbye')

    def getDirectory(self):
        self.saveDirectory = os.path.normpath(
            QFileDialog.getExistingDirectory(self, "Choose folder"))
        self.saveText.setText("Save to: " + str(self.saveDirectory))

    def getRemoteDirectory(self):
        user_home = os.path.expanduser("~")
        mint_network_path = f"{user_home}/.gvfs/"

        # If .gvfs isn't used, check the alternative system-wide mount location:
        if not os.path.exists(mint_network_path):
            # This path uses your Linux user ID (usually 1000)
            uid = os.getuid()
            mint_network_path = f"/run/user/{uid}/gvfs/"
        self.remote_repository = os.path.normpath(QFileDialog.getExistingDirectory(None, "Select Remote Directory", mint_network_path))
        print(f'Data will copied to {self.remote_repository}')

    def backup_data(self):
        print(f'Backing up recording to {self.remote_repository}')
        st = time.time()
        try:

            extra_args = ['--no-check-dest', '--ignore-checksum', '--contimeout=10s']
            rclone.copy(self.local_repository, self.remote_repository, extra_args)
            print(f'Recordings successfully backed up to {self.remote_repository} in {time.time()-st} seconds')
        except:
            print(f'Backup to {self.remote_repository} failed')

    def update_sources(self):
        self.cam_list = tools.list_cameras()
        self.flir_cam_list = tools.list_flir_cameras()
        self.thorcam_list = tools.list_thorlabs_cameras()
        for act in self.cam_menu.actions():
            self.cam_menu.removeAction(act)
        for dev in self.cam_list:
            new_action = QAction(dev, self)
            new_action.setCheckable(True)
            new_action.triggered.connect(self.update_active_cams)
            self.cam_menu.addAction(new_action)
        for dev in self.flir_cam_list:
            new_action = QAction(dev, self)
            new_action.setCheckable(True)
            new_action.triggered.connect(self.update_active_flircams)
            self.flircam_menu.addAction(new_action)
        for act in self.ipcam_menu.actions():
            self.ipcam_menu.removeAction(act)
        for dev in self.ip_list:
            new_action = QAction(dev, self)
            new_action.setCheckable(True)
            new_action.triggered.connect(self.update_active_ip_cams)
            self.ipcam_menu.addAction(new_action)
        self.port_list = tools.get_ports()
        for act in self.port_menu.actions():
            self.port_menu.removeAction(act)
        for port in self.port_list:
            print(port)
            new_action = QAction(port['name'], self)
            new_action.setCheckable(True)
            new_action.triggered.connect(self.update_active_ports)
            self.port_menu.addAction(new_action)
        for thorcam in self.thorcam_list:
            new_action = QAction(thorcam, self)
            new_action.setCheckable(True)
            new_action.triggered.connect(self.update_active_thorcams)
            self.thorcam_menu.addAction(new_action)
        
        self.arduino_triggered_thorcam = True
        if self.arduino_triggered_thorcam:
            self.thorcam_led_arduino_serial_num = '03535343335351903271'
            print(f'Looking for arduino with serial number {self.thorcam_led_arduino_serial_num} to control thorcam...')
            for port in self.port_list:
                if port['serial_number'] == self.thorcam_led_arduino_serial_num:
                    self.thorcam_arduino_path = port['id']
                    print(f'Arduino found at {self.thorcam_arduino_path}')
                    break
            else:
                print(f'Arduino not found, switching to free run mode')
                self.arduino_triggered_thorcam = False
      
            try:
                self.thorcam_arduino = serial.Serial(port=self.thorcam_arduino_path, baudrate=115200, timeout=1)
                self.thorcam_arduino.close()
                print(f'Arduino connected at {self.thorcam_arduino_path}')
            except:
                print(f'Arduino not found, switching to free run  mode')
                self.arduino_triggered_thorcam = False



    def update_active_cams(self):
        actions = self.cam_menu.actions()
        self.active_cams = {}
        for i, action in enumerate(actions):
            if action.isChecked():
                self.active_cams[action.text()] = i
        self.create_input_objects()
        print(self.active_cams)

    def update_active_flircams(self):
        actions = self.flircam_menu.actions()
        self.active_flircams = {}
        for i, action in enumerate(actions):
            if action.isChecked():
                self.active_flircams[action.text()] = i
        self.create_input_objects()
        print(self.active_flircams)

    def update_active_ip_cams(self):
        actions = self.ipcam_menu.actions()
        self.active_ipcams = {}
        for i, action in enumerate(actions):
            if action.isChecked():
                self.active_ipcams[action.text()] = action.text()
        self.create_input_objects()
        print(self.active_ipcams)

    def update_active_ports(self):
        actions = self.port_menu.actions()
        self.active_ports = {}
        for i, action in enumerate(actions):
            if action.isChecked():
                self.active_ports[action.text()] = action.text()
        self.create_input_objects()
        print(self.active_ports)

    def update_active_thorcams(self):
        actions = self.thorcam_menu.actions()
        self.active_thorcams = {}
        # Add actions to 
        for i, action in enumerate(actions):
            if action.isChecked():
                self.active_thorcams[action.text()] = action.text()
        self.create_input_objects()
        print(self.active_thorcams)

    def create_input_objects(self):
        # Generate tools.input_device objects for each active input device, set initial props
        self.cam_objects = {}
        for cam in self.active_cams:
            dev = self.cam_list[cam]
            print(dev)
            print(dev['info'].path)
            self.cam_objects[cam] = tools.input_device(
                name=dev['name'], device_type='usb_camera', device_path = dev['info'].path)
            self.cam_objects[cam].deviceID = self.active_cams[cam]
            self.cam_objects[cam].interval = 75
        for cam in self.active_flircams:
            dev = self.flir_cam_list[cam]
            self.cam_objects[cam] = tools.input_device(
                name=dev['name'], device_type='flir_camera', device_path = dev['info'].path)
            self.cam_objects[cam].deviceID = self.active_flircams[cam]
            self.cam_objects[cam].interval = 120
        for ipcam in self.active_ipcams:
            self.cam_objects[ipcam] = tools.input_device(
                name=ipcam, device_type='ip_camera')
            self.cam_objects[ipcam].deviceID = ipcam
            self.cam_objects[ipcam].interval = 120
        for port in self.active_ports:
            self.cam_objects[port] = tools.input_device(name=port, device_type='serial', device_path=port, calibration_factor = 1.0)
            for info in tools.get_ports():
                if info['name'] == port:
                    self.cam_objects[port].name = info['name']
                    self.cam_objects[port].device_path = info['id']
                    self.cam_objects[port].deviceID = info['id']
            if self.cam_objects[port].name == 'Force sensor':
                self.cam_objects[port].calibration_factor = -0.0038 ### calibration to read out in grams of force, needs empirical calibration
            self.cam_objects[port].interval = 10
        for thorcam in self.active_thorcams:
            self.cam_objects[thorcam] = tools.input_device(
                name=thorcam, device_type='thor_camera', parent=self)
            self.cam_objects[thorcam].deviceID = thorcam
            self.cam_objects[thorcam].interval = 100
      
        self.property_menus = {}
        # clear property menu
        for act in self.property_menu.actions():
            self.property_menu.removeAction(act)
        
        self.setter_acts = {}
        for name in self.cam_objects:
            # Create a menu for each data acquisition device
            obj = self.cam_objects[name]
            self.property_menus[name] = QMenu(name)
            # Create an item for each editiable prpoerty for each device
            for prop in obj.editable_properties:
                
                act = QAction(name + '--' + prop, self)
                act.triggered.connect(self.property_setter)
                self.property_menus[name].addAction(act)
          

            # add menu to property menu
            self.property_menu.addMenu(self.property_menus[name])
            

        self.init_display()

    def start_acq(self):
        if not len(self.cam_objects):
                    print('No devices selected!')
                    return
        self.acq_active = True
        start_time = time.time()
        dir_name = str(time.time())[0:10]
        self.trialFolder = os.path.join(self.saveDirectory, dir_name)
        self.trialFolder = os.path.normpath(self.trialFolder)
        os.makedirs(self.trialFolder)
        
        for cam in self.cam_objects:
            self.cam_objects[cam].parent = self
            self.cam_objects[cam].start(
                exp_dir=self.trialFolder, start_time=start_time, display=True, record=True)
        self.stop_action.setEnabled(True)
        self.preview_action.setEnabled(False)
        self.start_action.setEnabled(False)
        self.recording = True

    def start_preview(self):
        self.acq_active = True
        start_time = time.time()
        for cam in self.cam_objects:
            self.cam_objects[cam].parent = self
            self.cam_objects[cam].start(exp_dir = '/', start_time = start_time, display=True, record=False)
        self.stop_action.setEnabled(True)
        self.preview_action.setEnabled(False)
        self.start_action.setEnabled(False)
        self.recording = False

    def stop_acq(self):
        self.acq_active = False
        print(f'{self.recording=}')
        for cam in self.cam_objects:
            self.cam_objects[cam].stop()
        self.stop_action.setEnabled(False)
        self.start_action.setEnabled(True)
        self.preview_action.setEnabled(True)
        
        self.recording = False


    def init_display(self):
        self.GL.clear()
        self.cam_views = {}
        num_cams = len(self.cam_objects)
    
        r = 0
        c = 0
        font = pg.Qt.QtGui.QFont()
        font.setPixelSize(9)
        for cam in self.cam_objects:
            view = self.GL.addViewBox(r, c)
            if 'cam' in self.cam_objects[cam].device_type:
                view.invertY(False)
                im = pg.ImageItem(border='r', title = cam)
               #im.setImage(np.random.rand(4000,4000))
                hist_widget = pg.HistogramLUTItem(orientation='horizontal')
                hist_widget.setImageItem(im)
                view.addItem(im)
                view.addItem(hist_widget)
            else:
                view.invertY(True)
                im = pg.PlotItem(border='r', title = cam)
                im.getAxis('bottom').tickFont = font
                view.addItem(im)
                
          
            if c > 3:
                c = 0
                r = r+1
            else:
                c = c+1
            self.cam_views[cam] = im
            self.cam_objects[cam].view = im
        
        print('init display')
        for im in self.cam_views.values():
            pass
            # im.setImage(np.random.rand(100,100))
        # self.GL.addItem(self.GL)

    def set_save_dir():
        pass


if __name__ == '__main__':
    print('main loop')
    app = QApplication(sys.argv)
    ex = AcqGUI()
    sys.exit(app.exec_())
