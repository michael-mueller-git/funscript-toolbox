import os
import json
import copy
import sys
import time
import logging
import argparse

import numpy as np

from funscript_toolbox.data.ffmpegstream import FFmpegStream
from funscript_toolbox.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters
from funscript_toolbox.algorithms.videotracker import StaticVideoTracker, NoVideoTracker

class PositionAnnotation:

    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.video_file = os.path.abspath(args.input)
        self.video_info = FFmpegStream.get_video_info(args.input)
        self.annotation_file = ''.join(self.video_file.split('.')[:-1]) + ".json"
        self.load_annotation_file()
        self.stop = False
        self.ui = OpenCV_GUI(OpenCV_GUI_Parameters(
            video_info = self.video_info,
            skip_frames = 0,
            end_frame_number = self.video_info.length
        ))


    def load_annotation_file(self):
        if not os.path.exists(self.annotation_file):
            self.logger.info("Annotation file %s not exists", self.annotation_file)
            self.annotation = {
                'file': os.path.basename(self.video_file),
                'metadata': {
                    'fps': self.video_info.fps,
                    'height': self.video_info.height,
                    'width': self.video_info.width,
                    'length': self.video_info.length
                },
                "ffmpeg": "",
                'points': {}
            }
            return

        self.logger.info("Load existing Annotation file %s", self.annotation_file)
        with open(self.annotation_file, "r") as f:
            self.annotation = json.load(f)


    def save_annotation(self):
        self.logger.info("save annotation to %s", self.annotation_file)
        with open(self.annotation_file, "w") as f:
            json.dump(self.annotation, f, indent=4)


    def get_tracker(self, first_frame):
        selection = self.ui.menu(["1 Point", "2 Points"])

        preview_frame = copy.deepcopy(first_frame)
        bbox_top = self.ui.bbox_selector(
                preview_frame,
                "Select Top Tracking Feature",
                add_center = True
            )
        preview_frame = self.ui.draw_box_to_image(
                preview_frame,
                bbox_top,
                color=(255,0,255)
            )

        bbox_bottom = self.ui.bbox_selector(
                preview_frame,
                "Select Bottom Tracking Feature",
                add_center = True
            )

        tracker_top = StaticVideoTracker(
                        first_frame,
                        bbox_top,
                        self.video_info.fps
                    )

        if selection == 2:
            tracker_bottom = StaticVideoTracker(
                            first_frame,
                            bbox_bottom,
                            self.video_info.fps
                        )
        else:
            tracker_bottom = NoVideoTracker(bbox_bottom)

        return tracker_top, tracker_bottom


    def start(self):
        if len (self.annotation["points"]) > 0:
            self.logger.info("preview existing labels")
            self.preview()
            return

        self.logger.info("annotation not found, start annotation generator")
        first_frame = FFmpegStream.get_frame(self.video_file, 0)
        self.annotation["ffmpeg"] = self.ui.get_video_projection_config(first_frame, "vr_he_180_sbs")

        ffmpeg = FFmpegStream(
                video_path = self.video_file,
                config = self.annotation["ffmpeg"],
                skip_frames = 0,
                start_frame = 0
            )

        tracker = self.get_tracker(ffmpeg.read())

        tracking_result = []
        while ffmpeg.isOpen() and not self.stop:
            frame = ffmpeg.read()
            if frame is None:
                self.logger.warning("Failed to read next frame")
                break

            for i in range(2):
                tracker[i].update(frame)

            tracking_result.append([ tracker[i].result()[1] for i in range(2) ])

            key = self.ui.preview(
                    frame,
                    len(tracking_result),
                    texte = ["Press 'q' to stop tracking"],
                    boxes = tracking_result[-1]
                )

            if self.ui.was_key_pressed('q') or key == ord('q'):
                break

        ffmpeg.stop()

        point_distances = np.array([np.sqrt(np.sum((np.array(item[0][4:]) - np.array(item[1][4:])) ** 2, axis=0)) \
                    for item in tracking_result])

        max_distance_frame_number = np.argmax(np.array([abs(x) for x in point_distances]))

        selection = self.ui.menu(["Apply Offset by Dick Points", "Apply manual offset", "No Offset", "Discard"])

        if selection == 1:
            dick_pos = self.get_dick_pos(max_distance_frame_number)
            offset = (np.array(tracking_result[max_distance_frame_number][0][4:]) - dick_pos['top'], np.array(tracking_result[max_distance_frame_number][1][4:]) - dick_pos['bottom'])
            real_positions = [ [(np.array(tracking_result[i][0][4:]) - offset[0]).tolist(), (np.array(tracking_result[i][1][4:]) - offset[1]).tolist()] for i in range(len(tracking_result)) ]
        elif selection == 2:
            preview_frame = FFmpegStream.get_frame(self.video_file, max_distance_frame_number)
            preview_frame = FFmpegStream.get_projection(preview_frame, self.annotation["ffmpeg"])
            offset = [self.ui.get_point_offset(preview_frame, tracking_result[max_distance_frame_number][0][4:]), self.ui.get_point_offset(preview_frame, tracking_result[max_distance_frame_number][1][4:])]
            real_positions = [ [(np.array(tracking_result[i][0][4:]) - offset[0]).tolist(), (np.array(tracking_result[i][1][4:]) - offset[1]).tolist()] for i in range(len(tracking_result)) ]
        elif selection == 3:
            real_positions = [ [(np.array(tracking_result[i][0][4:])).tolist(), (np.array(tracking_result[i][1][4:])).tolist()] for i in range(len(tracking_result)) ]
        else:
            self.logger.warning("Discard new labels")
            return

        self.annotation["points"] = real_positions
        self.preview()
        self.save_annotation()


    def preview(self):
        ffmpeg = FFmpegStream(
                video_path = self.video_file,
                config = self.annotation["ffmpeg"],
                skip_frames = 0,
                start_frame = 0
            )

        frame_number = 0
        while ffmpeg.isOpen() and not self.stop:
            frame = ffmpeg.read()
            if frame_number >= len(self.annotation["points"]):
                break

            key = self.ui.preview(
                    frame,
                    frame_number,
                    texte = ["Press 'q' to stop preview"],
                    points = self.annotation["points"][frame_number]
                )

            time.sleep(0.05)

            frame_number += 1
            if self.ui.was_key_pressed('q') or key == ord('q'):
                break

        ffmpeg.stop()


    def get_dick_pos(self, max_distance_frame_number: int) -> dict:
        """ Get Start and End points of the dick

        Args:
            max_distance_frame_number (int): absolute frame number with max tracker distance

        Returns:
            dict: dick points
        """
        max_distance_frame = FFmpegStream.get_frame(self.video_file, max_distance_frame_number)
        max_distance_frame = FFmpegStream.get_projection(max_distance_frame, self.annotation["ffmpeg"])
        center_line = self.ui.line_selector(max_distance_frame, "draw line on center of dick")

        dick_pos = { 'top': np.array(center_line[1]), 'bottom': np.array(center_line[0]) } \
                if center_line[0][1] > center_line[1][1] \
                else { 'top': np.array(center_line[0]), 'bottom': np.array(center_line[1]) }

        return dick_pos



def setup_logging():
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', "INFO"),
        format='%(asctime)s %(levelname)s <%(filename)s:%(lineno)d> %(message)s',
        handlers=[
            logging.StreamHandler(stream=sys.stdout)
        ]
    )


def position_anotation_tool_entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type = str, help = "Video File")
    args = parser.parse_args()

    print("Tool WIP!!")

    setup_logging()

    if not os.path.exists(args.input):
        raise FileNotFoundError(args.input)

    position_annotation = PositionAnnotation(args)
    position_annotation.start()
