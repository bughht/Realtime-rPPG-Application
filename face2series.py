import threading
import time
from queue import Queue

import cv2 as cv
import dlib
import matplotlib.pyplot as plt
import numpy as np
import copy
import seaborn as sns

sns.set()


class CAM2FACE:
    def __init__(self) -> None:
        # get face detector and 68 face landmark
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            'data/shape_predictor_81_face_landmarks.dat')

        # get frontal camera of computer and get fps
        self.cam = cv.VideoCapture(0)
        # self.fps = self.cam.get(cv.CAP_PROP_FPS)
        self.fps = 20
        self.cam.set(cv.CAP_PROP_FPS, self.fps)

        # Initialize Queue for camera capture
        self.QUEUE_MAX = 256
        self.QUEUE_WINDOWS = 64
        self.Queue_rawframe = Queue(maxsize=3)
        # self.Queue_RGBhist_left = Queue(maxsize=self.QUEUE_MAX)
        # self.Queue_RGBhist_right = Queue(maxsize=self.QUEUE_MAX)
        # self.Queue_RGBhist_fore = Queue(maxsize=self.QUEUE_MAX)
        self.Queue_Sig_left = Queue(maxsize=self.QUEUE_MAX)
        self.Queue_Sig_right = Queue(maxsize=self.QUEUE_MAX)
        self.Queue_Sig_fore = Queue(maxsize=self.QUEUE_MAX)

        self.Queue_Time = Queue(maxsize=self.QUEUE_WINDOWS)

        self.Ongoing = False
        self.Flag_face = False
        self.Flag_Queue = False

        self.frame_display = None
        self.face_mask = None

        self.Sig_left = None
        self.Sig_right = None
        self.Sig_fore = None

    # Initialize process and start

    def PROCESS_start(self):
        self.Ongoing = True
        self.capture_process_ = threading.Thread(target=self.capture_process)
        self.roi_cal_process_ = threading.Thread(target=self.roi_cal_process)

        self.capture_process_.start()
        self.roi_cal_process_.start()

    # Process: capture frame from camera in specific fps of the camera
    def capture_process(self):
        while self.Ongoing:
            # time.sleep(0.02)
            # get frame
            self.ret, frame = self.cam.read()
            self.frame_display = copy.copy(frame)
            if self.Queue_Time.full():
                self.Queue_Time.get_nowait()
                self.fps = 1 / \
                    np.mean(np.diff(np.array(list(self.Queue_Time.queue))))

            # print(self.fps)
            # self.time = time_now
            if not self.ret:
                self.Ongoing = False
                break

            # check if rawframe queue is full, if true then clear the last data
            if self.Queue_rawframe.full():
                #print('Warning: Queue_rawframe full')
                self.Queue_rawframe.get_nowait()
            else:
                self.Queue_Time.put_nowait(time.time())

            try:
                self.Queue_rawframe.put_nowait(frame)
            except Exception as e:
                pass

    # Process: calculate roi from raw frame
    def roi_cal_process(self):
        while self.Ongoing:
            try:
                frame = self.Queue_rawframe.get_nowait()
            except Exception as e:
                # print(e)
                continue

            # get the roi of the frame (left/right)
            ROI_left, ROI_right, ROI_fore = self.ROI(frame)
            # check ROI exsistance
            if ROI_left is not None and ROI_right is not None and ROI_fore is not None:
                # produce rgb hist of mask (removed black)
                self.hist_left = self.RGB_hist(ROI_left)
                self.hist_right = self.RGB_hist(ROI_right)
                self.hist_fore = self.RGB_hist(ROI_fore)
                if self.Queue_Sig_left.full():
                    self.Sig_left = copy.copy(list(self.Queue_Sig_left.queue))
                    self.Queue_Sig_left.get_nowait()
                else:
                    self.Flag_Queue = False
                if self.Queue_Sig_right.full():
                    self.Sig_right = copy.copy(
                        list(self.Queue_Sig_right.queue))
                    self.Queue_Sig_right.get_nowait()
                else:
                    self.Flag_Queue = False
                if self.Queue_Sig_fore.full():
                    self.Sig_fore = copy.copy(list(self.Queue_Sig_fore.queue))
                    self.Queue_Sig_fore.get_nowait()
                    self.Flag_Queue = True
                else:
                    self.Flag_Queue = False
                # if self.Queue_RGBhist_left.full():
                #     self.Queue_RGBhist_left.get_nowait()
                # if self.Queue_RGBhist_right.full():
                #     self.Queue_RGBhist_right.get_nowait()
                # if self.Queue_RGBhist_fore.full():
                #     self.Queue_RGBhist_fore.get_nowait()

                # self.Queue_RGBhist_left.put(rgb_left)
                # self.Queue_RGBhist_right.put(rgb_right)
                # self.Queue_RGBhist_fore.put(rgb_fore)
                self.Queue_Sig_left.put_nowait(
                    self.Hist2Feature(self.hist_left))
                self.Queue_Sig_right.put_nowait(
                    self.Hist2Feature(self.hist_right))
                self.Queue_Sig_fore.put_nowait(
                    self.Hist2Feature(self.hist_fore))

            else:
                self.hist_left = None
                self.hist_right = None
                self.hist_fore = None
                # self.Queue_RGBhist_left.put(None)
                # self.Queue_RGBhist_right.put(None)
                # self.Queue_RGBhist_fore.put(None)
                self.Queue_Sig_left.queue.clear()
                self.Queue_Sig_right.queue.clear()
                self.Queue_Sig_fore.queue.clear()

    # Get the markpoint of the faces

    def Marker(self, img):
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.detector(img_gray)
        if len(faces) == 1:
            face = faces[0]
            landmarks = [[p.x, p.y]
                         for p in self.predictor(img, face).parts()]
            # for idx, point in enumerate(self.landmarks):
            #     pos = (point[0, 0], point[0, 1])
            #     cv.circle(img, pos, 2, color=(0, 255, 0))
        try:
            return landmarks
        except:
            return None

    # filter the image to ensure better performance
    def preprocess(self, img):
        return cv.GaussianBlur(img, (5, 5), 0)

    # Draw the ROI the image
    # ROI: left cheek and right cheek
    def ROI(self, img):
        img = self.preprocess(img)
        landmark = self.Marker(img)

        cheek_left = [1, 2, 3, 4, 48, 31, 28, 39]
        cheek_right = [15, 14, 14, 12, 54, 35, 28, 42]
        forehead = [69, 70, 71, 80, 72, 25, 24, 23, 22, 21, 20, 19, 18]

        mask_left = np.zeros(img.shape, np.uint8)
        mask_right = np.zeros(img.shape, np.uint8)
        mask_fore = np.zeros(img.shape, np.uint8)
        mask_display = np.zeros(img.shape, np.uint8)
        try:
            self.Flag_face = True
            pts_left = np.array(
                [landmark[i] for i in cheek_left], np.int32).reshape((-1, 1, 2))
            pts_right = np.array(
                [landmark[i] for i in cheek_right], np.int32).reshape((-1, 1, 2))
            pts_fore = np.array([landmark[i]
                                 for i in forehead], np.int32).reshape((-1, 1, 2))
            mask_left = cv.fillPoly(mask_left, [pts_left], (255, 255, 255))
            mask_right = cv.fillPoly(mask_right, [pts_right], (255, 255, 255))
            mask_fore = cv.fillPoly(
                mask_fore, [pts_fore], (255, 255, 255))

            # Erode Kernel: 30
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 30))
            mask_left = cv.erode(mask_left, kernel=kernel, iterations=1)
            mask_right = cv.erode(mask_right, kernel=kernel, iterations=1)
            mask_fore = cv.erode(
                mask_fore, kernel=kernel, iterations=1)
            # mask = cv.bitwise_or(mask_left, mask_right)
            mask_display_left, mask_display_right = copy.copy(
                mask_left), copy.copy(mask_right)
            mask_display_fore = copy.copy(mask_fore)

            mask_display_left[:, :, 1] = 0
            mask_display_right[:, :, 0] = 0
            mask_display_fore[:, :, 2] = 0

            mask_display = cv.bitwise_or(mask_display_left, mask_display_right)
            mask_display = cv.bitwise_or(mask_display, mask_display_fore)
            # mask_display = cv.fillPoly(mask_display,  [ pt = 0s_right], (0, 255, 0))
            self.face_mask = cv.addWeighted(mask_display, 0.25, img, 1, 0)

            ROI_left = cv.bitwise_and(mask_left, img)
            ROI_right = cv.bitwise_and(mask_right, img)
            ROI_fore = cv.bitwise_and(mask_fore, img)
            return ROI_left, ROI_right, ROI_fore

        except Exception as e:
            self.face_mask = img
            self.Flag_face = False
            return None, None, None

    # Cal hist of roi
    def RGB_hist(self, roi):
        b_hist = cv.calcHist([roi], [0], None, [256], [0, 256])
        g_hist = cv.calcHist([roi], [1], None, [256], [0, 256])
        r_hist = cv.calcHist([roi], [2], None, [256], [0, 256])
        b_hist = np.reshape(b_hist, (256))
        g_hist = np.reshape(g_hist, (256))
        r_hist = np.reshape(r_hist, (256))
        b_hist[0] = 0
        g_hist[0] = 0
        r_hist[0] = 0
        r_hist = r_hist/np.sum(r_hist)
        g_hist = g_hist/np.sum(g_hist)
        b_hist = b_hist/np.sum(b_hist)
        return [r_hist, g_hist, b_hist]

    def Hist2Feature(self, hist):
        hist_r = hist[0]
        hist_g = hist[1]
        hist_b = hist[2]

        # sgn_r = np.tanh(hist_r)
        # sgn_g = np.tanh(hist_g)
        # sgn_b = np.tanh(hist_b)

        hist_r /= np.sum(hist_r)
        hist_g /= np.sum(hist_g)
        hist_b /= np.sum(hist_b)

        dens = np.arange(0, 256, 1)
        mean_r = dens.dot(hist_r)
        mean_g = dens.dot(hist_g)
        mean_b = dens.dot(hist_b)

        return [mean_r, mean_g, mean_b]

    # Deconstruction

    def __del__(self):
        self.Ongoing = False
        self.cam.release()
        cv.destroyAllWindows()


if __name__ == '__main__':
    cam2roi = CAM2FACE()
    cam2roi.PROCESS_start()
    Hist_left_list = []
    Hist_right_list = []
    while True:
        print(cam2roi.fps)
    # time.sleep(1)
    # while True:
    # Hist_left = cam2roi.Queue_RGBhist_left.get()
    # Hist_right = cam2roi.Queue_RGBhist_right.get()
    # print(Hist_left)
    # cam2roi.__del__()
