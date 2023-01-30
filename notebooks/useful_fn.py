import cv2
from glob import glob
import pandas as pd

def show_basic_info(video):
    """
    Function to show basic information of the video.

    @video: path of the video.
    """
    data = cv2.VideoCapture(video)

    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)

    print(f"WIDTH: {data.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"HEIGHT: {data.get(cv2.CAP_PROP_FRAME_HEIGHT )}")
    print(f"FRAMES: {frames}")
    print(f"FPS: {(fps):0.2f}")
    print(f"TOTAL SECONDS: {(frames / fps):0.2f}") 

    data.release()

def generate_imgs(video, output, limit_frames=-1, every_n_frames=0):
    """
    This function generates images from video.

    @video: path of the video.
    @output: path of the folder for the output images.
    @limit_frames: limit of frames to output (default = -1 means that generates all images)
    @every_n_frames: steps of frames to generate (default = 0 means that generates all images)
    """
    stream = cv2.VideoCapture(video)
    n_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_idx in range(n_frames):
        ret, frame = stream.read()
        if not ret : break
        
        limit_condition = True if limit_frames == -1 else (frame_idx < limit_frames)
        every_n_frames_condition = True if every_n_frames == 0 else (frame_idx % every_n_frames == 0)

        if limit_condition and every_n_frames_condition:
            frame_no =  str(frame_idx).zfill(4)
            picture_path = f'{output}/frame_{frame_no}.jpeg'
            cv2.imwrite(picture_path, frame)

    stream.release()


def get_annotations(folder, width, height, categories_map, format_img="jpeg"):
    """
    This function is for generating the annotations dataframe to easy manipulate.

    @folder: folder of YOLO format data labels
    @width: width of images
    @height: height of images
    @categories_map: dictionary map for showing categories
    @format_img: format of images 
    """
    files = glob(folder + "/*.txt")
    ann_df = pd.DataFrame(columns=["label_name","center_x","center_y","bbox_width","bbox_height","image_name"])
    for file in files:
        image_name = file.split("/")[-1]
        image_name = image_name.replace("txt", format_img)
        with open(file) as f:
            lines = f.readlines()
        
        for line in lines:
            line_list = line.split(" ")
            line_list = [ l.strip() for l in line_list ]
            label = categories_map[int(line_list[0])]
            coord = line_list[1:]
            coord = [ float(c) for c in coord ]
            coord = [ c*width if coord.index(c) % 2 == 0 else c*height for c in coord ]
            
            new_row = [label] + coord + [image_name]
            ann_df.loc[len(ann_df.index)] = new_row
            
    return ann_df

