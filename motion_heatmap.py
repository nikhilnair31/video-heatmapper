import params
import numpy as np
import cv2
import copy
from make_video import make_video
from progress.bar import Bar

def init():
    global capture, background_subtractor, length, bar

    capture = cv2.VideoCapture('./input_videos/mall.mp4')
    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    bar = Bar('Processing Frames', max=length)
    print(f'length: {length}\n')

def do_first_frame_stuff():
    global first_frame, accum_image

    ret, frame = capture.read()
    params.height, params.width = frame.shape[:2]
    params.resized_height = int(params.height/params.resize_scale)
    params.resized_width = int(params.width/params.resize_scale)
    print(f'params.height: {params.height} params.width: {params.width}\n'
        f'params.resized_height: {params.resized_height} resized_width: {params.resized_width}\n')
    
    first_frame = copy.deepcopy(cv2.resize(frame,(params.resized_width, params.resized_height),fx=0,fy=0, interpolation = cv2.INTER_AREA))
    accum_image = np.zeros((params.resized_height, params.resized_width), np.uint8)

def main():
    global first_frame, accum_image, capture, background_subtractor, length, bar

    count = 0;
    init()
    do_first_frame_stuff()
    
    for i in range(0, length):
        ret, frame = capture.read()
        if (count % params.step_size == 0):
            frame = cv2.resize(frame,(params.resized_width, params.resized_height),fx=0,fy=0, interpolation = cv2.INTER_AREA)
            filter = background_subtractor.apply(frame)  # remove the background
            cv2.imwrite('./tmp/frame.jpg', frame)
            cv2.imwrite('./tmp/diff-bkgnd-frame.jpg', filter)

            ret, th1 = cv2.threshold(filter, params.threshold, params.maxValue, cv2.THRESH_BINARY)

            accum_image = cv2.add(accum_image, th1) # add to the accumulated image
            cv2.imwrite('./tmp/mask.jpg', accum_image)

            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)

            video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7, 0)

            name = f'./frames/frame{i}.jpg'
            cv2.imwrite(name, video_frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        count += 1
        bar.next()
    bar.finish()

    make_video('./frames/', './output_videos/output.avi')

    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)

    # save the final heatmap
    cv2.imwrite('./output_videos/diff-overlay.jpg', result_overlay)

    # cleanup
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
