from ultralytics import YOLO
import numpy as np
import cv2 as cv
import pandas as pd
import config as cfg
import utility as ut
import os

model_path = cfg.fileConfig.model_path
vid_path = ut.video_path(cfg.fileConfig.pitch1_name, 
                               cfg.fileConfig.pitcher_vids_path)
boxes_path = ut.csv_path_suffix(cfg.fileConfig.pitch1_name,
                                cfg.fileConfig.csv_path,
                                cfg.fileConfig.predictor_suffix)
confidence_ind = 4
num_extra_elems = 2
conf_threshold = 0.03

def get_boxes(model, vid_path: str, toi: tuple) -> dict:
    """
    Gets the bounding boxes from the video and returns a dictionary of boxes
    
    Args:
        model: YOLO model
        vid_path: Path to video
    
    Returns:
        boxes_dct: Dictionary of boxes
    
    Raises:
        ValueError: If the path is invalid
    """
    if not os.path.isfile(vid_path):
        raise ValueError("Invalid path")
    cap = cv.VideoCapture(vid_path)
    boxes_dct = {}
    count = 0
    while cap.isOpened():
        # read in frames
        _, image = cap.read()
        if image is None:
            break
        if toi[0] <= count <= toi[1]:
            results = model.predict(source=image, save=True, conf=conf_threshold)
            # Create boxes dictionary
            for i, box in enumerate(results[0].boxes.xyxy):  # For every box
                # Gets length of coordinates array
                len_coords_arr = len(box)
                # Creates a new array of coordinates with confidence
                coords_w_conf = np.zeros(len_coords_arr + num_extra_elems)
                # Populates the new array (apparently xyxy are tensors)
                for j, value in enumerate(box):
                    coords_w_conf[j] = value
                coords_w_conf[confidence_ind] = results[0].boxes.conf[i].item()
                coords_w_conf[-1] = int(i + 1)
                count = cap.get(cv.CAP_PROP_POS_FRAMES)
                dct_key = count + 0.01 * i
                boxes_dct[dct_key] = coords_w_conf

        if cv.waitKey(1) == ord("q"):
            break
        count+=1
    cap.release()
    cv.destroyAllWindows()
    return boxes_dct

def convert_boxes_df(dct: dict) -> pd.DataFrame():
    """
    Converts the boxes dictionary to a dataframe

    Args:
        dct: Dictionary of boxes
    
    Returns:
        df: Dataframe of boxes
    """
    df = pd.DataFrame.from_dict(dct, orient='index', columns=['x1', 'y1', 'x2',\
                                'y2', 'confidence', 'box_num'])
    #print(df)
    df.index.name = 'frame'
    df.reset_index(inplace=True)
    df = df.reindex(columns=['frame', 'box_num', 'x1', 'y1', 'x2', 'y2',
                            'confidence'])
    df['frame'] = df['frame'].astype(int)
    df['box_num'] = df['box_num'].astype(int)
    return df

def this_runner(model_path: str, vid_path: str, boxes_path: str) -> None:
    """
    Runner for the predictor code.

    Sets a YOLO model, tracks the video, gets the boxes, converts the boxes to
    a dataframe, and saves the dataframe to a csv.

    Args:
        model_path (str): Path to model
        vid_path (str): Path to video
        boxes_path (str): Path to save boxes to
    
    Returns:
        None
    """
    model = YOLO(model_path)
    if(cfg.fileConfig.release1_frame < 0):
        tup = ut.get_release_frame(vid_path)
        start_frame, pixel = tup
    toi = (start_frame, start_frame+\
           ut.pitch_time_frames(cfg.fileConfig.pitch1_velo))
    # Gets the boxes in a format unfit for a dataframe
    boxes_dct = get_boxes(model, vid_path, toi)
    # Converts the boxes to fittable format and writes to dataframe
    df = convert_boxes_df(boxes_dct)
    # Saves the dataframe to a csv
    #print(df)
    df.sort_values(by=['frame'], inplace=True)
    df.to_csv(boxes_path, index=False)

if __name__ == "__main__":
    this_runner(model_path, vid_path, boxes_path)

