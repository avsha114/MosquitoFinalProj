from roboflowoak import RoboflowOak
import cv2
import time


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.bg_color, 3, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.8, self.color, 1, self.line_type)

    def rectangle(self, frame, x1, y1, x2, y2):
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.bg_color, 6)
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)


if __name__ == '__main__':
    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="finalproj-hqv3h", confidence=0.3, overlap=0.5,
                     version="7", api_key="C8KHaS2KcTcQPxED81c0", rgb=True,
                     depth=True, device=None, blocking=True)
    stereo = rf.dai_pipe.stereo
    left = rf.dai_pipe.left
    right = rf.dai_pipe.right
    stereo.setLeftRightCheck(False)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(False)

    # Running our model and displaying the video output with detections
    while True:
        t0 = time.time()
        # The rf.detect() function runs the model inference
        result, frame, raw_frame, depth = rf.detect(visualize=True)
        predictions = result["predictions"]
        if predictions:
            predition_list = [p.json() for p in predictions]
            print("\nPredictions:")
            for p in predition_list:
                print(p)


        frame = cv2.resize(frame, (832, 832))  # 416*2


        text = TextHelper()

        for detection in predictions:
            dist = detection.depth

            det_min_x = int(detection.x - (detection.width/2)) * 2
            det_min_y = int(detection.y - (detection.height/2)) * 2

            text.putText(frame, "Distance: {:.2f} m".format(dist / 100), (det_min_x, det_min_y-5))

        cv2.imshow("preview", frame)
        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break