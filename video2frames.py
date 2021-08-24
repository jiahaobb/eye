import cv2, os
import numpy as np
import glob

def video2frame(video_path, frames_path):
    cap = cv2.VideoCapture(video_path) 
    try:
        os.mkdir(frames_path)
    except FileExistsError:
        pass
    frame_count = 0
    success = True
    while(success): 
        success, frame = cap.read() 
        try:
            frame_out = frame.transpose((1, 0, 2))
        except AttributeError:
            success = False
            break
        frame_out = np.rot90(frame_out)
        frame_out = np.rot90(frame_out)
        frame_out = np.rot90(frame_out)
        frame_out = np.fliplr(frame_out)
        cv2.imwrite("{}/{}.png".format(frames_path, str(frame_count).zfill(3)), frame_out) 
        frame_count = frame_count + 1

'''
    use numpy frames stored in list
'''
def frame2video_numpy(frames_list, video_path, fps=12, size=None):
    if size == None:
        first_img = frames_list[0]
        size = (first_img.shape[1], first_img.shape[0])
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for idx in range(0, len(frames_list)):
        frame = frames_list[idx]
        video_writer.write(frame)
    video_writer.release()

'''
    read video frames through frame paths
'''
def frame2video(frames_pth_list, video_path, fps=12, size=None):
    if size == None:
        first_img = cv2.imread(frames_pth_list[0])
        size = (first_img.shape[1], first_img.shape[0])
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for idx in range(0, len(frames_pth_list)):
        item = frames_pth_list[idx]
        frame = cv2.imread(item)
        video_writer.write(frame)
    video_writer.release()

if __name__ == "__main__":
    # frames to video
    # frames_root = '/Volumes/jiangywq/data0108'
    # videos_root = '/Users/jiangyingwenqi/Do wnloads/data0108'
    # # session_names = ['cup_finger2', 'wyj_finger2']
    # session_names = ['wyj_finger2']
    # session_pths = [os.path.join(frames_root, x) for x in session_names]
    # video_pths = [os.path.join(videos_root, x) for x in session_names]
    # my_fps = 60
    # for (data_pth, video_pth) in zip(session_pths, video_pths):
    #     if not os.path.exists(data_pth):
    #         print('{} does not exists.'.format(data_pth))
    #         continue
    #     for cam_idx in range(1, 9):
    #         video_path = os.path.join(video_pth, "cam{}.mp4".format(cam_idx))
    #         frames_name = 'image.cam0{}_*.png'.format(cam_idx)
    #         frames_pth_list = glob.glob(os.path.join(data_pth, "*", frames_name))
    #         frames_pth_list.sort(key=lambda x: int(x.split(os.path.sep)[-2]))
    #         print("data_pth: {}".format(data_pth))
    #         frame2video(frames_pth_list=frames_pth_list, video_path=video_path, fps=my_fps)
    #         print("video_path: {} done".format(video_path))

    # video to frames
    frames_dir = '/Users/jiangyingwenqi/Desktop/data_1203/frames'
    videos_dir = '/Users/jiangyingwenqi/Desktop/data_1203'
    videos_pths = [os.path.join(videos_dir, "data3_cam{}.mp4".format(x)) for x in range(1,9)]
    frames_pths = [os.path.join(frames_dir, "{}".format(x)) for x in range(1,9)]
    for videos_pth, frames_pth in zip(videos_pths, frames_pths):
        video2frame(videos_pth, frames_pth)
