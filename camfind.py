from cv2_enumerate_cameras import enumerate_cameras
cams = enumerate_cameras();
print(cams)
for camera in cams:
    print(f'{camera.index}: {camera.name}')