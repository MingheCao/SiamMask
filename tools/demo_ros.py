import glob
from tools.test import *

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
parser.add_argument('--config', dest='config', default='/home/rislab/Workspace/SiamMask/experiments/siammask_sharp/config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
args = parser.parse_args()

global cv_img
def image_callback(data):
    global cv_img

    bridge = CvBridge()
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")


if __name__ == '__main__':
    rospy.init_node('siam_node', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, image_callback)
    rate = rospy.Rate(10)

    #Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    siammask = load_pretrain(siammask, '/home/rislab/Workspace/SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth')
    siammask.eval().to(device)

    global cv_img
    im=cv_img
    while np.shape(im) == 0:
        print(np.shape(im))
        im=cv_img

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', im, False, False)
        x, y, w, h = init_rect
    except:
        exit()

    im=cv_img
    target_pos = np.array([x + w / 2, y + h / 2])
    target_sz = np.array([w, h])
    state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker

    toc = 0
    while not rospy.is_shutdown():
        tic = cv2.getTickCount()

        im=cv_img
        state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
        location = state['ploygon'].flatten()
        mask = state['mask'] > state['p'].seg_thr

        im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
        cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
        cv2.imshow('SiamMask', im)
        key = cv2.waitKey(1)
        if key > 0:
            break

        rate.sleep()

        toc += cv2.getTickCount() - tic
    # toc /= cv2.getTickFrequency()
    # # fps = f / toc
    # # print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
