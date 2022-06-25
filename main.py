from roboflowoak import RoboflowOak
import cv2
import time

# This class should create text in our format:
class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    # We use putText to write text on our frame using cv2 library:
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type)


class FPSHandler:
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()

        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    def fps(self):
        return self.frame_cnt / (self.timestamp - self.start)


if __name__ == '__main__':
    model_name = "finalproj-hqv3h"
    model_version = "7"
    private_api_key = "C8KHaS2KcTcQPxED81c0"
    confidence_level = 0.3


    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model=model_name, confidence=confidence_level,
                     version=model_version, api_key=private_api_key, rgb=True, depth=True)

    base_resolution = rf.resolution
    scale_factor = 1.5

    # Apply settings for better inference from long distances
    stereo = rf.dai_pipe.stereo
    left = rf.dai_pipe.left
    right = rf.dai_pipe.right
    stereo.setLeftRightCheck(False)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    # Running our model and displaying the video output with detections

    fps = FPSHandler()

    while True:
        # The rf.detect() function runs the model inference
        result, frame, raw_frame, depth = rf.detect(visualize=True)
        predictions = result["predictions"]

        fps.next_iter()

        # Print predictions to console
        if predictions:
            predition_list = [p.json() for p in predictions]
            print("\nPredictions:")
            for p in predition_list:
                print(p)

        # Resize frame for better view
        frame = cv2.resize(frame, (int(base_resolution * scale_factor), int(base_resolution * scale_factor)))  # 416*2

        text = TextHelper()

        for detection in predictions:
            dist = detection.depth

            det_min_x = int((detection.x - (detection.width/2)) * scale_factor)
            det_min_y = int((detection.y - (detection.height/2)) * scale_factor)

            text.putText(frame, "Distance: {:.2f} m".format(dist / 100), (det_min_x, det_min_y-5))

        fps_x = int((base_resolution * scale_factor) - (0.25 * (base_resolution * scale_factor)))
        fps_y = int((base_resolution * scale_factor) - (0.05 * (base_resolution * scale_factor)))
        text.putText(frame, "{:.2f} FPS".format(fps.fps()), (fps_x, fps_y))

        cv2.imshow("preview", frame)
        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break