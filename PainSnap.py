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
import PainSnap_tools as tools
import time
import os
import serial
from serial.tools import list_ports
from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, OPERATION_MODE

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
        self.port_menu = QMenu('Select serial devices')
        self.property_menu = QMenu('Set device properties...')

        # self.update_sources()

        self.start_action = QAction('Acquire', self)
        self.stop_action = QAction('End acquisition', self)
        self.stop_action.setEnabled(False)
        self.find_action = QAction('Find inputs', self)
        self.add_ip_action = QAction('Add ip camera...', self)
        self.remove_ip_action = QAction('Remove ip camera', self)

        self.find_action.triggered.connect(self.update_sources)
        self.start_action.triggered.connect(self.start_acq)
        self.stop_action.triggered.connect(self.stop_acq)

        self.set_path_action = QAction('Set save path...', self)
        self.set_path_action.triggered.connect(self.getDirectory)

        self.config_menu.addAction(self.set_path_action)
        self.config_menu.addAction(self.find_action)
        self.config_menu.addMenu(self.cam_menu)
        self.config_menu.addMenu(self.thorcam_menu)
        self.config_menu.addMenu(self.ipcam_menu)
        self.config_menu.addMenu(self.port_menu)
        self.config_menu.addMenu(self.property_menu)
        self.update_sources()

        self.acq_menu.addAction(self.start_action)
        self.acq_menu.addAction(self.stop_action)

        self.saveDirectory = os.path.normpath(r'/home/rlab-scope/Data/PainSnap data')
        self.saveText = QLineEdit("Save to: " + self.saveDirectory, self)
        self.saveText.setReadOnly(True)
        self.saveText.resize(1600, 50)
        self.saveText.move(0, 850)
        self.saveText.setStyleSheet("""QLineEdit { background-color
                                    : transparent ; color: blue}""")
        self.active_cams = {}
        self.active_ipcams = {}
        self.active_ports = {}
        self.active_thorcams = {}   
        self.running_thorcams = []
        self.cam_objects = {}
        self.cam_views = {}
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
        d, ok = QInputDialog.getText(self, 'Device properties', f'Set {prop} for {dev}:')
        try:
            value = eval(d)
        except:
            value = d
        setattr(self.cam_objects[dev], prop, value)


    def closeEvent(self,event):
        for thorcam in self.running_thorcams:
            print(f'Disconnecting from ThorLabs Camera {thorcam}')
            thorcam.dispose()
        else:
            print('Goodbye')

    def getDirectory(self):
        self.saveDirectory = os.path.normpath(
            QFileDialog.getExistingDirectory(self, "Choose folder"))
        self.saveText.setText("Save to: " + str(self.saveDirectory))

    def update_sources(self):
        self.cam_list = tools.list_cameras()
        self.thorcam_list = tools.list_thorlabs_cameras()
        print(self.cam_list)
        for act in self.cam_menu.actions():
            self.cam_menu.removeAction(act)
        for dev in self.cam_list:
            new_action = QAction(dev, self)
            new_action.setCheckable(True)
            new_action.triggered.connect(self.update_active_cams)
            self.cam_menu.addAction(new_action)
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
            new_action = QAction(port['id'], self)
            new_action.setCheckable(True)
            new_action.triggered.connect(self.update_active_ports)
            self.port_menu.addAction(new_action)
        for thorcam in self.thorcam_list:
            new_action = QAction(thorcam, self)
            new_action.setCheckable(True)
            new_action.triggered.connect(self.update_active_thorcams)
            self.thorcam_menu.addAction(new_action)

    def update_active_cams(self):
        actions = self.cam_menu.actions()
        self.active_cams = {}
        for i, action in enumerate(actions):
            if action.isChecked():
                self.active_cams[action.text()] = i
        self.create_input_objects()
        print(self.active_cams)

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
        for i, action in enumerate(actions):
            if action.isChecked():
                self.active_thorcams[action.text()] = action.text()
        self.create_input_objects()
        print(self.active_thorcams)

    def create_input_objects(self):
        # Generate tools.input_device objects for each active input device, set initial props
        self.cam_objects = {}
        for cam in self.active_cams:
            self.cam_objects[cam] = tools.input_device(
                name=cam, device_type='usb_camera')
            self.cam_objects[cam].deviceID = self.active_cams[cam]
            self.cam_objects[cam].interval = 20
        for ipcam in self.active_ipcams:
            self.cam_objects[ipcam] = tools.input_device(
                name=ipcam, device_type='ip_camera')
            self.cam_objects[ipcam].deviceID = ipcam
            self.cam_objects[ipcam].interval = 110
        for port in self.active_ports:
            self.cam_objects[port] = tools.input_device(
                name=port, device_type='serial')
            self.cam_objects[port].deviceID = port
            self.cam_objects[port].interval = 10
        for thorcam in self.active_thorcams:
            self.cam_objects[thorcam] = tools.input_device(
                name=thorcam, device_type='thor_camera')
            self.cam_objects[thorcam].deviceID = thorcam
            self.cam_objects[thorcam].interval = 50
        self.property_menus = {}
        # clear property menu
        for act in self.property_menu.actions():
            self.property_menu.removeAction(act)
        
        self.setter_acts = {}
        for name in self.cam_objects:
            # TODO
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
        self.acq_active = True
        start_time = time.time()
        dir_name = str(time.time())[0:10]
        self.trialFolder = os.path.join(self.saveDirectory, dir_name)
        self.trialFolder = os.path.normpath(self.trialFolder)
        os.makedirs(self.trialFolder)
        for cam in self.cam_objects:
            self.cam_objects[cam].start(
                exp_dir=self.trialFolder, start_time=start_time)
        self.stop_action.setEnabled(True)
        self.start_action.setEnabled(False)

    def stop_acq(self):
        self.acq_active = False
        for cam in self.cam_objects:
            self.cam_objects[cam].stop()
        self.stop_action.setEnabled(False)
        self.start_action.setEnabled(True)

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
