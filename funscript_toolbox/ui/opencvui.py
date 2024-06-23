import cv2
import os
import copy
import yaml
import time
import logging
import pynput.keyboard

from queue import Queue
from datetime import datetime
from funscript_toolbox.data.ffmpegstream import FFmpegStream, VideoInfo
from dataclasses import dataclass

import numpy as np


@dataclass
class OpenCV_GUI_Parameters:
    video_info: VideoInfo
    skip_frames: int
    end_frame_number: int
    preview_scaling: float = 0.6
    text_start_x: int = 30
    text_start_y: int = 30
    text_line_height: int = 30
    font_size: float = 0.75
    fps_smoothing_factor: int = 100
    window_name_prefix: str = "OpenCV-UI"
    text_border_width: int = 6


class KeypressHandler:
    """ Keypress Handler for OpenCV GUI """

    def __init__(self):
        self.keypress_queue = Queue(maxsize=32)
        self.listener = pynput.keyboard.Listener(
            on_press = self.on_key_press,
            on_release = None
        )
        self.listener.start()


    def stop_keypress_handler(self):
        try:
            self.listener.stop()
        except:
            pass


    def __del__(self):
        self.stop_keypress_handler()


    def on_key_press(self, key: pynput.keyboard.Key) -> None:
        """ Our key press handle to register the key presses

        Args:
            key (pynput.keyboard.Key): the pressed key
        """
        if not self.keypress_queue.full():
            self.keypress_queue.put(key)


    def clear_keypress_queue(self) -> None:
        """ Clear the key press queue """
        while self.keypress_queue.qsize() > 0:
            self.keypress_queue.get()


    def was_any_accept_key_pressed(self) -> bool:
        """ Check if 'space' or 'enter' was presssed

        Returns:
            bool: True if an accept key was pressed else False
        """
        while self.keypress_queue.qsize() > 0:
            if any('{0}'.format(self.keypress_queue.get()) == x for x in ["Key.space", "Key.enter"]):
                return True

        return False


    def was_key_pressed(self, key: str) -> bool:
        """ Check if key was presssed

        Args:
            key (str): the key to check

        Returns:
            bool: True if 'q' was pressed else False
        """
        if key is None or len(key) == 0:
            return False

        while self.keypress_queue.qsize() > 0:
            if '{0}'.format(self.keypress_queue.get()) == "'"+key[0]+"'":
                return True

        return False


class DrawSingleLineWidget(object):
    def __init__(self, background_img, window_name, preview_scaling, color=(36,255,12)):
        self.original_image = background_img
        self.clone = self.original_image.copy()
        self.window_name = window_name
        self.preview_scaling = preview_scaling
        self.color = color

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.extract_coordinates)

        self.start_coordinate = None
        self.end_coordinate = None

    def extract_coordinates(self, event, x, y, flags, parameters):
        x = round(x/self.preview_scaling)
        y = round(y/self.preview_scaling)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_coordinate = (x,y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.end_coordinate = (x,y)
            self.clone = self.original_image.copy()
            cv2.line(self.clone, self.start_coordinate, self.end_coordinate, self.color, 2)

    def show_image(self):
        return self.clone

    def get_result(self):
        return [ self.start_coordinate, self.end_coordinate ]


class OpenCV_GUI(KeypressHandler):
    """ High Level OpenCV GUI.

    Args:
        params (OpenCV_GUI_Parameters): configuration parameters for OpenCV GUI
    """

    def __init__(self, params: OpenCV_GUI_Parameters):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        projection_config_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "projection.yaml")
        if not os.path.exists(projection_config_file):
            raise FileNotFoundError(f"File {projection_config_file} not exists")

        with open(projection_config_file) as f:
            self.projection_config = yaml.load(f, Loader = yaml.FullLoader)

        self.params = params
        self.preview_fps = []
        self.fps_timer = cv2.getTickCount()
        self.preview_image_origin_height = 0
        self.preview_image_origin_width = 0
        self.window_name = "{} - {}".format(self.params.window_name_prefix, datetime.now().strftime("%H:%M:%S"))
        self.__reset_print_positions()
        self.preview_scaling_applied = False
        self.preview_image = None
        self.preview_image_without_scale = None


    def __del__(self):
        super().stop_keypress_handler()
        super().__del__()
        self.close()


    def close(self):
        """Close all OpenCV GUIs"""
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass


    def __reset_print_positions(self):
        """ Reset all print positions """
        self.text_y_pos = {
                'left': self.params.text_start_y,
                'center': self.params.text_start_y,
                'column2': self.params.text_start_y,
                'right': self.params.text_start_y
            }


    def __determine_preview_scaling(self) -> None:
        """ Determine the scaling for current monitor setup """
        scale = []
        try:
            from screeninfo import get_monitors
            for monitor in get_monitors():
                if monitor.width > monitor.height:
                    scale.append(
                        min([
                            monitor.width / float(self.preview_image_origin_width),
                            monitor.height / float(self.preview_image_origin_height)
                        ])
                    )
        except:
            pass

        if len(scale) == 0:
            self.logger.error("Monitor resolution info not found")
            self.monitor_preview_scaling = 1.0
        else:
            # asume we use the largest monitor for scipting
            self.monitor_preview_scaling = self.params.preview_scaling * max(scale)
            self.monitor_preview_scaling = self.monitor_preview_scaling / float(os.getenv("QT_SCALE_FACTOR", 1))


    def set_background_image(self, image: np.ndarray, copy_image: bool = False) -> None:
        """ Set the preview image

        Args:
            image (np.ndarray): opencv image
            copy_image (bool): create an copy of the image
        """
        if image is None:
            image = np.full((720, 1240, 3), 0, dtype=np.uint8)

        self.preview_image = copy.deepcopy(image) if copy_image else image

        if self.preview_image.shape[0] != self.preview_image_origin_height or self.preview_image.shape[1] != self.preview_image_origin_width:
            self.preview_image_origin_height = self.preview_image.shape[0]
            self.preview_image_origin_width = self.preview_image.shape[1]
            self.__determine_preview_scaling()

        self.preview_scaling_applied = False
        self.__reset_print_positions()


    def draw_box(self, bbox, color: tuple = (255, 0, 255), connect=False) -> None:
        """ Draw an tracking box to the preview image

        Args:
            bbox (tuple): tracking box with (x,y,w,h)
            color (tuple): RGB color values for the box
            connect (bool): Connect boxes by a line (only if center point is available in box!)
        """
        assert self.preview_image is not None

        if not isinstance(bbox, list):
            bbox = [bbox]

        for box in bbox:
            if box and len(box) >= 4:
                cv2.rectangle(
                        self.preview_image,
                        (box[0], box[1]),
                        ((box[0] + box[2]), (box[1] + box[3])),
                        color,
                        1,
                        1
                    )
                if len(box) >= 6:
                    cv2.circle(
                            self.preview_image,
                            (box[4], box[5]),
                            4,
                            color,
                            2
                        )

        if len(bbox) == 2 and connect:
            if len(bbox[0]) >= 6 and len(bbox[1]) >= 6:
                cv2.line(self.preview_image, (bbox[0][4], bbox[0][5]), (bbox[1][4], bbox[1][5]), color, 2)


    @staticmethod
    def draw_box_to_image(image: np.ndarray, bbox, color: tuple = (255, 0, 255)) -> np.ndarray:
        """ Draw an tracking box to given image

        Args:
            image (np.ndarray): image
            bbox (tuple or list): tracking box with (x,y,w,h) or list of tracking boxes
            color (tuple): RGB color values for the box

        Returns:
            np.ndarray: opencv image
        """
        if not isinstance(bbox, list):
            bbox = [bbox]

        for box in bbox:
            if box and len(box) >= 4:
                cv2.rectangle(
                        image,
                        (box[0], box[1]),
                        ((box[0] + box[2]), (box[1] + box[3])),
                        color,
                        3,
                        1
                    )
                if len(box) >= 6:
                    cv2.circle(
                            image,
                            (box[4], box[5]),
                            4,
                            color,
                            2
                        )

        return image


    @staticmethod
    def draw_point_to_image(
            image: np.ndarray,
            point,
            color: tuple = (255, 0, 255),
            connect_points: bool = False
            ) -> np.ndarray:
        """ Draw an point to given image

        Args:
            image (np.ndarray): image
            point (tuple or list): points (x,y) or list of points
            color (tuple): RGB color values for the box
            connect_points (bool): connect points with an line

        Returns:
            np.ndarray: opencv image
        """
        if not isinstance(point, list):
            point = [point]

        for p in point:
            cv2.circle(
                    image,
                    (p[0], p[1]),
                    4,
                    color,
                    2
                )

        if len(point) > 1 and connect_points:
            for i in range(len(point)-1):
                cv2.line(
                        image,
                        point[i],
                        point[i+1],
                        color,
                        2)

        return image


    def get_preview_fps(self) -> float:
        """ Get current processing FPS

        Returns
            float: FPS
        """
        if len(self.preview_fps) < 1:
            return 1.0

        return np.mean((
            self.preview_fps[-self.params.fps_smoothing_factor:] \
            if len(self.preview_fps) < self.params.fps_smoothing_factor \
            else self.preview_fps
        ))


    def print_fps(self) -> None:
        """ Draw processing FPS to the preview image """
        assert self.preview_image is not None
        self.print_text(str(int(self.get_preview_fps())) + ' fps')


    def __update_processing_fps(self) -> None:
        """ Update processing FPS """
        self.preview_fps.append((self.params.skip_frames+1) * cv2.getTickFrequency() / (cv2.getTickCount()-self.fps_timer))
        self.fps_timer = cv2.getTickCount()


    def print_time(self, current_frame_number: int) -> None:
        """ Draw Time on the preview image

        Args:
            current_frame_number (int): current absolute frame number
        """
        assert self.preview_image is not None
        current_timestamp = FFmpegStream.frame_to_timestamp(current_frame_number, self.params.video_info.fps)
        current_timestamp = ''.join(current_timestamp[:-4])

        if self.params.end_frame_number < 1:
            end_timestamp = FFmpegStream.frame_to_timestamp(self.params.video_info.length, self.params.video_info.fps)
            end_timestamp = ''.join(end_timestamp[:-4])
        else:
            end_timestamp = FFmpegStream.frame_to_timestamp(self.params.end_frame_number, self.params.video_info.fps)
            end_timestamp = ''.join(end_timestamp[:-4])

        txt = current_timestamp + ' / ' + end_timestamp
        self.print_text(txt, text_position_x = 'right')


    def print_text(self, txt, color: tuple = (0,0,255), text_position_x: str = 'left') -> None:
        """ Draw text to an image/frame

        Args:
            txt (str, list): the text to plot on the image
            colot (tuple): BGR Color tuple
            text_position_x (str): text position ['left',  'right', 'center', 'column2']
        """
        assert self.preview_image is not None
        assert text_position_x in self.text_y_pos.keys()

        if not isinstance(txt, list):
            txt = [txt]

        for line in txt:
            if text_position_x.lower() == 'left':
                x = self.params.text_start_x
            elif text_position_x.lower() == 'right':
                (text_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.params.font_size, 2)
                x = max([0, int(self.preview_image.shape[1] - self.params.text_start_x - text_w) ])
            elif text_position_x.lower() == 'center':
                (text_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.params.font_size, 2)
                x = max([0, round((self.preview_image_origin_width / 2) - (text_w / 2))])
            elif text_position_x.lower() == 'column2':
                x = round(self.preview_image_origin_width / 2 + self.params.text_start_x)
            else:
                raise NotImplementedError("Print Text at position %s is not implemented", text_position_x)

            (text_w, text_h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, self.params.font_size, 2)
            cv2.rectangle(
                    self.preview_image,
                    (x - self.params.text_border_width, self.text_y_pos[text_position_x] - self.params.text_border_width),
                    (x + text_w + self.params.text_border_width, self.text_y_pos[text_position_x] + text_h + self.params.text_border_width),
                    (0, 0, 0),
                    -1
                )
            cv2.putText(
                    self.preview_image,
                    str(line),
                    (x, round(self.text_y_pos[text_position_x] + text_h + self.params.font_size - 1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.params.font_size,
                    color,
                    2
                )

            self.text_y_pos[text_position_x] += self.params.text_line_height


    def scale_preview_image(self)-> None:
        """ Scale image for preview """
        assert self.preview_image is not None

        if self.preview_scaling_applied:
            return

        self.preview_image_without_scale = copy.deepcopy(self.preview_image)
        self.preview_image = cv2.resize(
                self.preview_image,
                None,
                fx=self.monitor_preview_scaling,
                fy=self.monitor_preview_scaling
            )

        self.preview_scaling_applied = True


    def show(self, wait: int = 1) -> int:
        """ Show annotated preview image

        Args:
            wait (int): waitKey time in millisecondscv2.resizeWindow("Resized_Window", 300, 700)

        Returns:
            int: waitKey result
        """
        assert self.preview_image is not None
        self.scale_preview_image()
        cv2.imshow(self.window_name, self.preview_image)
        return cv2.waitKey(wait)


    def selectROI(self) -> tuple:
        """ OpenCV selectROI wrapper

        Returns:
            tuple: bbox (x,y,w,h)
        """
        assert self.preview_image is not None
        self.scale_preview_image()
        return cv2.selectROI(self.window_name, self.preview_image, False)


    def show_loading_screen(self, txt: str = "Please wait...", background_size=None) -> None:
        """ Show an loading screen

        Args:
            txt (str): text to display
            image_size (tuple): optional tuple with (h,w,c)
        """
        if background_size is not None:
            self.set_background_image(np.full(background_size, 0, dtype=np.uint8))
            self.print_text(txt, color=(0,0,255))
            self.show()

        elif self.preview_image_without_scale is not None:
            self.set_background_image(np.full(self.preview_image_without_scale.shape, 0, dtype=np.uint8))
            self.print_text(txt, color=(0,0,255))
            self.show()



    def line_selector(self,
            image: np.ndarray,
            txt: str) -> list:
        """ Line Selector Widget

        Args:
            image (np.ndarray): background image
            txt (str): text to display

        Returns:
            list: start and endpoint of line
        """
        line_widget = DrawSingleLineWidget(image, self.window_name, self.monitor_preview_scaling)
        note = ""
        while True:
            self.set_background_image(line_widget.show_image(), copy_image=False)
            self.print_text(txt)
            self.print_text("Press 'space' to continue")
            if len(note) > 0:
                self.print_text(note)

            ret = self.show(5)

            if self.was_any_accept_key_pressed() or any(ret == x for x in [ord(' '), 13]):
                result = line_widget.get_result()
                if result[0] is not None and result[1] is not None:
                    if abs(result[0][0] - result[1][0]) + abs(result[0][1] - result[1][1]) > 10:
                        self.show_loading_screen()
                        return result
                    else:
                        note = "ERROR: Invalid selection"
                else:
                    note = "ERROR: Missing Input"


    def min_max_selector(self,
            image_min: np.ndarray,
            image_max: np.ndarray,
            info: str = "",
            title_min: str = "",
            title_max: str = "",
            recommend_lower: int = 0,
            recommend_upper: int = 99) -> tuple:
        """ Min Max selection Window

        Args:
            image_min (np.ndarray): the frame/image with lowest position
            image_max (np.ndarray): the frame/image with highest position
            info (str): additional info string th show on the Window
            title_min (str): title for the min selection
            title_max (str): title for the max selection
            recommend_lower (int): recommend lower value
            recommend_upper (int): recommend upper value

        Returns:
            tuple: with selected (min: flaot, max float)
        """
        image = np.concatenate((image_min, image_max), axis=1)

        self.set_background_image(image, copy_image=True)
        self.show(1)

        cv2.createTrackbar("Min", self.window_name, recommend_lower, 99, lambda _: None)
        cv2.createTrackbar("Max", self.window_name, recommend_upper, 99, lambda _: None)

        self.clear_keypress_queue()
        trackbarValueMin = recommend_lower
        trackbarValueMax = recommend_upper
        self.logger.info("Waiting for user input")

        while True:
            try:
                self.set_background_image(image, copy_image=True)
                self.print_text(title_min if title_min != "" else "Min")
                self.print_text(title_max if title_max != "" else "Max", text_position_x='column2')
                self.print_text("Set {} to {}".format('Min', trackbarValueMin))
                self.print_text("Set {} to {}".format('Max', trackbarValueMax), text_position_x='column2')
                self.print_text("Info: " + info)
                self.print_text("Press 'space' to continue", text_position_x='column2')
                ret = self.show(25)

                if self.was_any_accept_key_pressed() or any(ret == x for x in [ord(' '), 13]):
                    break

                trackbarValueMin = cv2.getTrackbarPos("Min", self.window_name)
                trackbarValueMax = cv2.getTrackbarPos("Max", self.window_name)
            except:
                pass

        self.logger.info("Receive User Input")
        self.show_loading_screen()

        return (trackbarValueMin, trackbarValueMax) \
                if trackbarValueMin < trackbarValueMax \
                else (trackbarValueMax, trackbarValueMin)


    @staticmethod
    def get_center(box: tuple) -> tuple:
        """ Get the cencter point of an box

        Args:
            box (tuple): the predicted bounding box

        Returns:
            tuple (x,y) of the current point
        """
        return ( round(box[0] + box[2]/2), round(box[1] + box[3]/2) )


    def bbox_selector(self, image: np.ndarray, txt: str, add_center: bool = False) -> tuple:
        """ Window to get an bounding box from user input

        Args:
            image (np.ndarray): opencv image e.g. the first frame to determine the bounding box
            txt (str): additional text to display on the selection window
            add_center (bool): add center cordinates to the box

        Returns:
            tuple: user input bounding box tuple (x,y,w,h)
        """
        self.set_background_image(image, copy_image=True)
        self.print_text("Select area with Mouse and Press 'space' to continue")
        self.print_text(txt)

        while True:
            bbox = self.selectROI()

            if bbox is None or len(bbox) == 0:
                continue

            if bbox[0] == 0 or bbox[1] == 0 or bbox[2] < 9 or bbox[3] < 9:
                continue

            break

        # revert the preview scaling
        bbox = (
                round(bbox[0]/self.monitor_preview_scaling),
                round(bbox[1]/self.monitor_preview_scaling),
                round(bbox[2]/self.monitor_preview_scaling),
                round(bbox[3]/self.monitor_preview_scaling)
            )

        if add_center:
            center = self.get_center(bbox)
            bbox = (bbox[0], bbox[1], bbox[2], bbox[3], center[0], center[1])

        self.logger.info("User Input: %s", str(bbox))
        return bbox


    def get_video_projection_config(self, image :np.ndarray, projection: str, show_keys: bool = True) -> dict:
        """ Get the video projection config form user input

        Args:
            image (np.ndarray): opencv vr 180 or 360 image
            projection (str): projection key from config
            show_keys (bool): show key shortcuts

        Returns:
            dict: projection config
        """
        assert projection in self.projection_config.keys()
        config = copy.deepcopy(self.projection_config[projection])

        h, w = image.shape[:2]
        if self.projection_config[projection]['parameter']['height'] == -1:
            scaling = config['parameter']['width'] / float(w)
            config['parameter']['height'] = round(h * scaling)
        elif self.projection_config[projection]['parameter']['width'] == -1:
            scaling = config['parameter']['height'] / float(h)
            config['parameter']['width'] = round(w * scaling)

        # NOTE: improve processing speed to make this menu more responsive
        if image.shape[0] > 6000 or image.shape[1] > 6000:
            image = cv2.resize(image, None, fx=0.25, fy=0.25)

        if image.shape[0] > 3000 or image.shape[1] > 3000:
            image = cv2.resize(image, None, fx=0.5, fy=0.5)

        if "mouse" in config and config["mouse"] == True:
            self.logger.info("Show Mouse ROI Menu")
            preview = FFmpegStream.get_projection(image, config)
            selected_bbox = self.bbox_selector(preview, "Select ROI")
            config['parameter']['x'] = selected_bbox[0]
            config['parameter']['y'] = selected_bbox[1]
            config['parameter']['w'] = selected_bbox[2]
            config['parameter']['h'] = selected_bbox[3]
            self.show_loading_screen()
            return config

        ui_texte = {}
        if "keys" in config.keys():
            for param in config['keys'].keys():
                if param in config['parameter'].keys() and all(item in config["keys"][param].keys() for item in ["increase", "decrease"]):
                    ui_texte[param] = "Use '{}', '{}' to increase/decrease {} = ${{val}}".format(
                        config["keys"][param]["increase"],
                        config["keys"][param]["decrease"],
                        param
                    )

        self.clear_keypress_queue()

        self.logger.info("Show ROI Menu")
        if len(ui_texte) > 0:
            # we need an user input
            parameter_changed, selected = True, False
            while not selected:
                if parameter_changed:
                    parameter_changed = False
                    preview = FFmpegStream.get_projection(image, config)
                    self.set_background_image(preview)
                    if show_keys:
                        self.print_text("Press 'space' to use current viewpoint")
                        self.print_text("Press '0' (NULL) to reset view")
                        final_ui_texte = [ui_texte[k].replace('${val}', str(config['parameter'][k])) for k in ui_texte.keys()]
                        self.print_text(final_ui_texte)

                ret = self.show()
                if ret in [ord(' '), 13]:
                    break

                while self.keypress_queue.qsize() > 0:
                    pressed_key = '{0}'.format(self.keypress_queue.get())
                    if pressed_key == "Key.space" or pressed_key == "Key.enter":
                        selected = True
                        break

                    if pressed_key == "'0'":
                        config = copy.deepcopy(self.projection_config[projection])
                        if config['parameter']['height'] == -1:
                            scaling = config['parameter']['width'] / float(w)
                            config['parameter']['height'] = round(h * scaling)
                        elif config['parameter']['width'] == -1:
                            scaling = config['parameter']['height'] / float(h)
                            config['parameter']['width'] = round(w * scaling)
                        parameter_changed = True
                        break

                    if "keys" not in config.keys():
                        break

                    for param in config['keys'].keys():
                        if param in config['parameter'].keys() and all(x in config["keys"][param].keys() for x in ["increase", "decrease"]):
                            if pressed_key == "'" + config["keys"][param]["increase"] + "'":
                                config['parameter'][param] += 5
                                parameter_changed = True
                                break
                            elif pressed_key == "'" + config["keys"][param]["decrease"] + "'":
                                config['parameter'][param] -= 5
                                parameter_changed = True
                                break


        self.show_loading_screen()
        del config["keys"]
        return config



    def preview(self,
            image: np.ndarray,
            current_frame_number: int = 0,
            texte: list = [],
            boxes: list = [],
            points: list = [],
            wait: int = 1,
            show_fps: bool = True,
            show_time: bool = True
            ) -> int:
        """
        Args:
            image (np.ndarray): image to preview
            current_frame_number (int): current frame number
            texte (list, optional): list of texte to annotate the preview image
            boxes (list, optional): draw boxes on the preview image list of list with boxes
            wait (int): waitKey delay in milliseconds
            show_fps (bool): show processing fps
            show_time (bool): show processing time

        Returns:
            int: waitKey result
        """
        self.set_background_image(image)
        self.__update_processing_fps()

        if show_time:
            self.print_time(current_frame_number)

        if show_fps:
            self.print_fps()

        self.print_text(texte)
        for boxes_item in boxes:
            self.draw_box(boxes_item, connect=True)
        if len(points) > 0:
            self.preview_image = self.draw_point_to_image(self.preview_image, points)

        return self.show(wait)


    def menu(self,
            title,
            menu_items: list) -> int:

        """ Show menu and get selection

        Args:
            title(str): Title to show
            menu_items (list): Menu items to display

        Returns:
            int selected menu entry
        """
        if self.preview_image_without_scale is not None:
            self.set_background_image(np.full(self.preview_image_without_scale.shape, 0, dtype=np.uint8))
        else:
            self.set_background_image(np.full((512,512,3), 0, dtype=np.uint8))
        self.print_text(title, color=(0,0,255))
        self.print_text("Type number to select menu entry:", color=(0,0,255))
        for idx, item in enumerate(menu_items):
            self.print_text("[" + str(idx+1) + "] " + item, color=(0,0,255))

        ret = 0
        while True:
            ret = self.show()
            if ret in [ord(str(x)) for x in range(1,len(menu_items)+1)]:
                break

        return int(chr(ret))


    def get_point_offset(self, frame, start_point) -> tuple:
        """ Get an offset value for an start point by user input

        Args:
            frame (np.ndarray): frame
            start_point (tuple): start point

        Returns:
            tuple: offset in (x,y)
        """
        parameter_changed= True
        new_position = copy.deepcopy(start_point)
        accept = False
        self.clear_keypress_queue()
        time.sleep(0.5)
        self.clear_keypress_queue()
        while not accept:
            if parameter_changed:
                parameter_changed = False
                preview = copy.deepcopy(frame)
                preview = self.draw_point_to_image(preview, new_position)
                self.set_background_image(preview)
                self.print_text("Press space to confirm", color=(0,0,255))

            ret = self.show()
            # if ret in [ord(' '), 13]:
            #     print("accept #1")
            #     break

            while self.keypress_queue.qsize() > 0:
                pressed_key = '{0}'.format(self.keypress_queue.get())
                if pressed_key == "Key.space" or pressed_key == "Key.enter":
                    print("accept #2")
                    accept = True
                    break

                if pressed_key == "'w'":
                    parameter_changed = True
                    new_position = (new_position[0], new_position[1]-5)
                elif pressed_key == "'a'":
                    parameter_changed = True
                    new_position = (new_position[0]-5, new_position[1])
                elif pressed_key == "'s'":
                    parameter_changed = True
                    new_position = (new_position[0], new_position[1]+5)
                elif pressed_key == "'d'":
                    parameter_changed = True
                    new_position = (new_position[0]+5, new_position[1])

        return (start_point[0] - new_position[0], start_point[1] - new_position[1])


    def get_rectangle_offset(self, frame, rectangle) -> tuple:
        """ Get an offset value for an start point by user input

        Args:
            frame (np.ndarray): frame
            rectangle (tuple): rectangle in format (x1,y,x2,y2)

        Returns:
            tuple: rectangle (x1,y,x2,y2)
        """
        parameter_changed= True
        new_position = copy.deepcopy(rectangle)
        accept = False
        self.clear_keypress_queue()
        time.sleep(0.1)
        point = 1
        while not accept:
            if parameter_changed:
                parameter_changed = False
                preview = copy.deepcopy(frame)
                boxes = [(new_position[0], new_position[1], new_position[2]-new_position[0], new_position[3]-new_position[1])]
                preview= self.draw_box_to_image(preview, boxes)
                if point == 1:
                    preview = self.draw_point_to_image(preview, (new_position[0], new_position[1]))
                else:
                    preview = self.draw_point_to_image(preview, (new_position[2], new_position[3]))
                self.set_background_image(preview)
                self.print_text("Press space to confirm", color=(0,0,255))

            ret = self.show()
            # if ret in [ord(' '), 13]:
            #     print("accept #1")
            #     point += 1
            #     parameter_changed = True
            #     self.clear_keypress_queue()
            #     time.sleep(0.5)
            #     self.clear_keypress_queue()
            #     if point > 2:
            #         break

            while self.keypress_queue.qsize() > 0:
                pressed_key = '{0}'.format(self.keypress_queue.get())
                if pressed_key == "Key.space" or pressed_key == "Key.enter":
                    print("accept #2")
                    point += 1
                    parameter_changed = True
                    self.clear_keypress_queue()
                    time.sleep(0.5)
                    self.clear_keypress_queue()
                    if point > 2:
                        accept = True
                        break

                if pressed_key == "'w'":
                    parameter_changed = True
                    if point == 1:
                        new_position = (new_position[0], new_position[1]-5, new_position[2], new_position[3])
                    else:
                        new_position = (new_position[0], new_position[1], new_position[2], new_position[3]-5)
                elif pressed_key == "'a'":
                    parameter_changed = True
                    if point == 1:
                        new_position = (new_position[0]-5, new_position[1], new_position[2], new_position[3])
                    else:
                        new_position = (new_position[0], new_position[1], new_position[2]-5, new_position[3])
                elif pressed_key == "'s'":
                    parameter_changed = True
                    if point == 1:
                        new_position = (new_position[0], new_position[1]+5, new_position[2], new_position[3])
                    else:
                        new_position = (new_position[0], new_position[1], new_position[2], new_position[3]+5)
                elif pressed_key == "'d'":
                    parameter_changed = True
                    if point == 1:
                        new_position = (new_position[0]+5, new_position[1], new_position[2], new_position[3])
                    else:
                        new_position = (new_position[0], new_position[1], new_position[2]+5, new_position[3])

        return (new_position[0] - rectangle[0], new_position[1] - rectangle[1], new_position[2] - rectangle[2], new_position[3] - rectangle[3])
