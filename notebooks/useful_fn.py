import cv2
from glob import glob
import pandas as pd
import os

def get_basic_info(video):
    """
    Function to show basic information of the video.

    @video: path of the video.
    """
    data = cv2.VideoCapture(video)

    frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = data.get(cv2.CAP_PROP_FPS)

    info = {}
    info["width"] = int(data.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["height"] = int(data.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info["frames"] = int(frames)
    info["fps"] = fps
    info["total_seconds"] = frames / fps

    data.release()

    return info

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

def replace_class_in_yolo_file(folder, replace_map):
    """
    Function to replace missclasses in Yolo files using a map.
    This function creates a folder output with the results in the same directory.

    @folder: where the Yolo files are located.
    @replace_map: dictionary used to replace, for example: { 0:2, 1:4 }
    """
    files_path = os.listdir(folder)
    try:
        os.mkdir(f"{folder}/output")
    except:
        pass

    validation_data = []

    for file_path in  files_path:
        with open(f"{folder}/{file_path}") as f:
            data = f.readlines()

        data = [ d.strip() for d in data ]
        new_data = []
        for line in data:
            new_line = line.split(" ")
            new_line[0] = str(replace_map[int(new_line[0])])
            new_line = " ".join(new_line)
            new_data.append(new_line)

        with open(f"{folder}/output/{file_path}", 'w') as f:
            for d in new_data:
                f.write(d)
                if d != new_data[-1] : f.write("\n")

        # check data
        with open(f"{folder}/output/{file_path}") as f:
            data_check = f.readlines()

        val = (len(data_check) == len(data))
        if val:
            data = [ l.strip().split(" ") for l in data ]
            data = [ " ".join(l) for l in data ]

            data_check = [ l.strip().split(" ") for l in data_check ]
            data_check = [ " ".join(l) for l in data_check ]

            zip_data = list(zip(data, data_check))
            # just must be diff for the first character (class)
            val = all(e[0][1:] == e[1][1:] for e in zip_data)

        print(f"Old data: {data}")
        print(f"New data: {data_check}")

        validation_data.append(val)
    
    if all(v for v in validation_data):
        print("DATA FINISHED SUCCESSFULLY")
    else:
        print("DATA WAS CORRUPTED")
    
