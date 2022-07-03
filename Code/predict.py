import time
import cv2
import numpy as np
import subprocess

from retinaface import Retinaface

if __name__ == "__main__":
    retinaface = Retinaface()
    mode = "video"
    video_path      = "img/vid.mp4"
    audio_path      = 'img/vid.mp3'
    video_save_path = "vidrec.mp4"
    video_fps       = 25.0
    test_interval   = 100

    def video2mp3(file_name):

        outfile_name = file_name.split('.')[0] + '.mp3'
        cmd = 'ffmpeg -i ' + file_name + ' -f mp3 ' + outfile_name
        print(cmd)
        subprocess.call(cmd, shell=True)


    def video_add_mp3(file_name, mp3_file):
        outfile_name = file_name.split('.')[0] + '-f.mp4'
        subprocess.call('ffmpeg -i ' + file_name
                        + ' -i ' + mp3_file + ' -strict -2 -f mp4 '
                        + outfile_name, shell=True)

    video2mp3(file_name= video_path)

    if mode == "video":

        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("error")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # detecting 
            frame = np.array(retinaface.detect_image(frame))
            
            # RGBtoBGR meets the opencv display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)      
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")

        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

        # Add sound to video
        video_add_mp3(file_name= video_save_path, mp3_file=audio_path)

