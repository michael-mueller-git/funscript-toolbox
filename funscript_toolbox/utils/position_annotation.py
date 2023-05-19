import os
import json
import copy
import sys
import logging
import argparse

from funscript_toolbox.data.ffmpegstream import FFmpegStream
from funscript_toolbox.ui.opencvui import OpenCV_GUI, OpenCV_GUI_Parameters
from funscript_toolbox.algorithms.videotracker import StaticVideoTracker

class PositionAnnotation:

    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.video_file = args.input
        self.video_info = FFmpegStream.get_video_info(args.input)
        self.annotation_file = ''.join(args.input.split('.')[:-1])
        self.load_annotation_file()
        self.stop = False
        self.ui = OpenCV_GUI(OpenCV_GUI_Parameters(
            video_info = self.video_info,
            skip_frames = 0,
            end_frame_number = self.video_info.length
        ))


    def load_annotation_file(self):
        if not os.path.exists(self.annotation_file):
            self.logger.info("Annotation file not exists")
            self.annotation = {
                'file': os.path.basename(self.video_file),
                'metadata': {
                    'fps': self.video_info.fps,
                    'height': self.video_info.height,
                    'width': self.video_info.width,
                    'length': self.video_info.length
                },
                "ffmpeg": "",
                'positons': {}
            }
            return

        self.logger.info("Load existing Annotation file %s", self.annotation_file)
        with open(self.annotation_file, "r") as f:
            self.annotation = json.load(f)


    def save_annotation(self):
        with open(self.annotation_file, "w") as f:
            json.dump(self.annotation, f, indent=4)


    def get_tracker(self, first_frame):
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

        tracker_bottom = StaticVideoTracker(
                        first_frame,
                        bbox_bottom,
                        self.video_info.fps
                    )

        return tracker_top, tracker_bottom


    def start(self):
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
