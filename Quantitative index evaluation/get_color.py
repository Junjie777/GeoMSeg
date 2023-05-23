import cv2, os


def capture_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = [str(image[y, x, i]) for i in range(3)]
        print('[' + ','.join(color) + '],')


if __name__ == '__main__':
    image = cv2.imread(r'')

    window = 'click and get color'
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, capture_event)
    while True:
        cv2.imshow(window, image)
        key = cv2.waitKey(1)
        if key == 13 or key == 27:
            break
    cv2.destroyAllWindows()