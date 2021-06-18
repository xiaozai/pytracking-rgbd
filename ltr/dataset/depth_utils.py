import cv2
import numpy as np
import matplotlib.pyplot as plt

class p_config(object):
    grabcut_extra = 50
    grabcut_rz_threshold = 300
    grabcut_rz_factor = 1.5
    minimun_target_pixels = 16
    grabcut_iter = 3
    radius = 500

def sigmoid(dp, mu=None, sigma=4, U=1.8, L=0.2):
    dp = dp / 1000
    dp = np.nan_to_num(dp, nan=0)
    dp_values = dp[np.nonzero(dp)]

    if mu is None:
        # mu = np.median(dp)
        mu = np.median(dp_values)

    upper = U * mu
    lower = L * mu

    # dp[dp > upper] = upper
    # dp[dp < lower] = lower
    dp[dp > upper] = 0
    dp[dp < lower] = 0

    img = 1 / (1 + np.exp((dp-mu) / sigma))

    # colormap
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
    img = np.asarray(img*255, dtype=np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    return img

def get_layered_image_by_depth(depth_image, target_depth, dtype='centered_colormap'):

    p = p_config()

    if target_depth is not None:
        low = max(target_depth-p.radius, 0)
        high = target_depth + p.radius

        layer = depth_image.copy()
        layer[layer < low] = high + 10
        layer[layer > high] = high + 10
    else:
        layer = depth_image.copy()

    layer = remove_bubbles(layer, bubbles_size=200)

    if dtype == 'centered_colormap':
        layer = cv2.normalize(layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        layer = np.asarray(layer, dtype=np.uint8)
        layer = cv2.applyColorMap(layer, cv2.COLORMAP_JET)
    elif dtype == 'centered_norm_depth':
        layer = cv2.normalize(layer, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        layer = np.asarray(layer, dtype=np.uint8)
        layer = cv2.merge((layer, layer, layer))
    elif dtype == 'centered_raw_depth':
        layer = np.asarray(layer)
        layer = np.stack((layer, layer, layer), axis=2)

    return layer

def remove_bubbles(image, bubbles_size=100):
    try:
        binary_map = (image > 0).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        mask = np.zeros((image.shape), dtype=np.uint8)
        for i in range(0, nb_components):
            if sizes[i] >= bubbles_size:
                mask[output == i+1] = 1

        if len(image.shape)>2:
            image = image * mask[:, :, np.newaxis]
        else:
            image = image * mask
    except:
        pass

    return image

def get_target_mask(depth, target_box):
    '''
        To estimate the target depth by using cv2.grabCut
    '''

    p = p_config()

    H, W = depth.shape

    target_box = [int(bb) for bb in target_box]
    x0, y0, w0, h0 = target_box
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x0+w0, W)
    y1 = min(y0+h0, H)
    possible_target = depth[y0:y1, x0:x1]
    median_depth = np.median(possible_target) + 10

    # bubbles_size = int(target_box[2]*target_box[3]*0.1)
    try:
        ''' add the surrounding extra pixels as the background '''
        extra_y0 = max(y0 - p.grabcut_extra, 0)
        extra_x0 = max(x0 - p.grabcut_extra, 0)
        extra_y1 = min(y1 + p.grabcut_extra, H)
        extra_x1 = min(x1 + p.grabcut_extra, W)

        ''' new coordinates in cropped patch with extra pixels'''
        rect_x0 = x0 - extra_x0
        rect_y0 = y0 - extra_y0
        rect_x1 = min(rect_x0 + w0, extra_x1)
        rect_y1 = min(rect_y0 + h0, extra_y1)
        rect = [rect_x0, rect_y0, rect_x1-rect_x0, rect_y1-rect_y0]

        target_patch = depth[extra_y0:extra_y1, extra_x0:extra_x1]
        target_patch = np.nan_to_num(target_patch, nan=np.max(target_patch))

        ''' filter depth image, then convert to colormap '''
        image = target_patch.copy()
        image[image>median_depth*2] = median_depth*2 # !!!!!!!!!!
        image[image < 10] = median_depth*2

        '''To downsample the target_patch in order to speed up the cv2.grabCut'''
        i_H, i_W = image.shape
        rz_factor = p.grabcut_rz_factor if min(i_W, i_H) > p.grabcut_rz_threshold else 1
        rect_rz = [int(rt//rz_factor) for rt in rect]
        rz_dim = (int(i_W//rz_factor), int(i_H//rz_factor))

        image = cv2.resize(image, rz_dim, interpolation=cv2.INTER_AREA)
        # image = remove_bubbles(image, bubbles_size=bubbles_size)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        ''' 0-pixels and 2-pixels are background(set to 0), 1-pixels and 3-pixels are foreground(set to 1)'''
        mask_init = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(image, mask_init, rect_rz, bgdModel, fgdModel, p.grabcut_iter, cv2.GC_INIT_WITH_RECT)
        mask = np.where((mask_init==2)|(mask_init==0),0,1).astype('uint8')
        # mask = remove_bubbles(mask, bubbles_size=bubbles_size)
        mask = cv2.resize(mask, (i_W, i_H), interpolation=cv2.INTER_AREA)

        '''crop'''
        # mask = mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        ''' padding'''
        depth_masks = np.zeros_like(depth)
        depth_masks[extra_y0:extra_y1, extra_x0:extra_x1] = mask

    except:
        depth_masks = np.ones_like(depth)

    # plt.subplot(111)
    # plt.imshow(mask)
    # plt.show(block=False)

    return depth_masks

def get_target_depth_and_mask(depth, target_box):

    '''
        To estimate the target depth by using cv2.grabCut
    '''

    p = p_config()

    H, W = depth.shape

    target_box = [int(bb) for bb in target_box]
    x0, y0, w0, h0 = target_box
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x0+w0, W)
    y1 = min(y0+h0, H)
    possible_target = depth[y0:y1, x0:x1]
    median_depth = np.median(possible_target) + 10

    bubbles_size = int(target_box[2]*target_box[3]*0.1)
    try:
        ''' add the surrounding extra pixels as the background '''
        extra_y0 = max(y0 - p.grabcut_extra, 0)
        extra_x0 = max(x0 - p.grabcut_extra, 0)
        extra_y1 = min(y1 + p.grabcut_extra, H)
        extra_x1 = min(x1 + p.grabcut_extra, W)

        rect_x0 = x0 - extra_x0
        rect_y0 = y0 - extra_y0
        rect_x1 = min(rect_x0 + w0, extra_x1)
        rect_y1 = min(rect_y0 + h0, extra_y1)
        rect = [rect_x0, rect_y0, rect_x1-rect_x0, rect_y1-rect_y0]

        target_patch = depth[extra_y0:extra_y1, extra_x0:extra_x1]
        target_patch = np.nan_to_num(target_patch, nan=np.max(target_patch))

        ''' filter depth image, then convert to colormap '''
        image = target_patch.copy()
        image[image>median_depth*2] = median_depth*2 # !!!!!!!!!!
        image[image < 10] = median_depth*2

        '''To downsample the target_patch in order to speed up the cv2.grabCut'''
        i_H, i_W = image.shape
        rz_factor = p.grabcut_rz_factor if min(i_W, i_H) > p.grabcut_rz_threshold else 1
        rect_rz = [int(rt//rz_factor) for rt in rect]
        rz_dim = (int(i_W//rz_factor), int(i_H//rz_factor))

        image = cv2.resize(image, rz_dim, interpolation=cv2.INTER_AREA)
        image = remove_bubbles(image, bubbles_size=bubbles_size)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        ''' 0-pixels and 2-pixels are background(set to 0), 1-pixels and 3-pixels are foreground(set to 1)'''
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(image, mask, rect_rz, bgdModel, fgdModel, p.grabcut_iter, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        mask2 = remove_bubbles(mask2, bubbles_size=bubbles_size)
        mask2 = cv2.resize(mask2, (i_W, i_H), interpolation=cv2.INTER_AREA)


        depth_masks = np.zeros_like(depth, dtype=np.uint8)
        depth_masks[extra_y0:extra_y1, extra_x0:extra_x1] = mask2

        ''' to get the target depth values '''
        target_pixels = target_patch * mask2
        target_pixels = target_pixels.flatten()
        target_pixels.sort()
        target_pixels = target_pixels[target_pixels>0]

        if len(target_pixels) > p.minimun_target_pixels:
            hist, bin_edges = np.histogram(target_pixels, bins=20)
            peak_idx = np.argmax(hist)
            selected_target_pixels = target_pixels
            target_depth_low = bin_edges[peak_idx]
            target_depth_high= bin_edges[peak_idx+1]
            selected_target_pixels = selected_target_pixels[selected_target_pixels<=target_depth_high]
            selected_target_pixels = selected_target_pixels[selected_target_pixels>=target_depth_low]
            target_depth = np.median(selected_target_pixels)
        else:
            target_depth = median_depth
            depth_masks = np.ones_like(depth)
    except:
        target_depth = median_depth
        depth_masks = np.ones_like(depth)

    # print('Traget depth : ', target_depth)
    return target_depth, np.asarray(depth_masks, dtype=np.float32)

def get_target_depth(depth, target_box):

    '''
        To estimate the target depth by using cv2.grabCut
    '''

    p = p_config()

    H, W = depth.shape

    target_box = [int(bb) for bb in target_box]
    x0, y0, w0, h0 = target_box
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x0+w0, W)
    y1 = min(y0+h0, H)
    possible_target = depth[y0:y1, x0:x1]
    median_depth = np.median(possible_target) + 10

    bubbles_size = int(target_box[2]*target_box[3]*0.1)
    try:
        ''' add the surrounding extra pixels as the background '''
        extra_y0 = max(y0 - p.grabcut_extra, 0)
        extra_x0 = max(x0 - p.grabcut_extra, 0)
        extra_y1 = min(y1 + p.grabcut_extra, H)
        extra_x1 = min(x1 + p.grabcut_extra, W)

        rect_x0 = x0 - extra_x0
        rect_y0 = y0 - extra_y0
        rect_x1 = min(rect_x0 + w0, extra_x1)
        rect_y1 = min(rect_y0 + h0, extra_y1)
        rect = [rect_x0, rect_y0, rect_x1-rect_x0, rect_y1-rect_y0]

        target_patch = depth[extra_y0:extra_y1, extra_x0:extra_x1]
        target_patch = np.nan_to_num(target_patch, nan=np.max(target_patch))

        ''' filter depth image, then convert to colormap '''
        image = target_patch.copy()
        image[image>median_depth*2] = median_depth*2 # !!!!!!!!!!
        image[image < 10] = median_depth*2

        '''To downsample the target_patch in order to speed up the cv2.grabCut'''
        i_H, i_W = image.shape
        rz_factor = p.grabcut_rz_factor if min(i_W, i_H) > p.grabcut_rz_threshold else 1
        rect_rz = [int(rt//rz_factor) for rt in rect]
        rz_dim = (int(i_W//rz_factor), int(i_H//rz_factor))

        image = cv2.resize(image, rz_dim, interpolation=cv2.INTER_AREA)
        image = remove_bubbles(image, bubbles_size=bubbles_size)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = np.asarray(image, dtype=np.uint8)
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)

        ''' 0-pixels and 2-pixels are background(set to 0), 1-pixels and 3-pixels are foreground(set to 1)'''
        mask = np.zeros(image.shape[:2], np.uint8)
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(image, mask, rect_rz, bgdModel, fgdModel, p.grabcut_iter, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        mask2 = remove_bubbles(mask2, bubbles_size=bubbles_size)
        mask2 = cv2.resize(mask2, (i_W, i_H), interpolation=cv2.INTER_AREA)


        ''' to get the target depth values '''
        target_pixels = target_patch * mask2
        target_pixels = target_pixels.flatten()
        target_pixels.sort()
        target_pixels = target_pixels[target_pixels>0]

        if len(target_pixels) > p.minimun_target_pixels:
            hist, bin_edges = np.histogram(target_pixels, bins=20)
            peak_idx = np.argmax(hist)
            selected_target_pixels = target_pixels
            target_depth_low = bin_edges[peak_idx]
            target_depth_high= bin_edges[peak_idx+1]
            selected_target_pixels = selected_target_pixels[selected_target_pixels<=target_depth_high]
            selected_target_pixels = selected_target_pixels[selected_target_pixels>=target_depth_low]
            target_depth = np.median(selected_target_pixels)
        else:
            target_depth = median_depth
    except:
        target_depth = median_depth

    # print('Traget depth : ', target_depth)
    return target_depth

if __name__ == '__main__':

    import os

    data_path = '/home/sgn/Data1/yan/Datasets/EstimatedDepth/LaSOT/'
    sequences = os.listdir(data_path)

    for seq in sequences:
        print(seq)
        if os.path.isdir(os.path.join(data_path, seq)):
            depth_path = os.path.join(data_path, seq, 'depth')
            gt_path = os.path.join(data_path, seq, 'groundtruth.txt')

            with open(gt_path, 'r') as fp:
                bboxes = fp.readlines()
            bboxes = [box.strip() for box in bboxes]

            num_frames = len(os.listdir(depth_path))

            depth_values = np.zeros(num_frames)

            for ii in range(1, num_frames+1):
                frame_path = os.path.join(depth_path, '%08d.png'%ii)
                if os.path.isfile(frame_path):
                    depth = cv2.imread(frame_path, -1)
                    depth[depth>10000] = 0

                    box = bboxes[ii-1]
                    box = [int(float(b)) for b in box.split(',')]

                    target_depth = get_target_depth(depth, box)
                    depth_values[ii-1] = target_depth

                    print(seq, ii-1, target_depth)

                    img = sigmoid(depth, mu=target_depth)

                    colormap = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
                    colormap = np.asarray(colormap, dtype=np.uint8)
                    colormap = cv2.applyColorMap(colormap, cv2.COLORMAP_JET)

                    cv2.imshow('img', img)
                    cv2.waitKey(1)

            # out_path = os.path.join(data_path, seq, 'target_depth.txt')
            # with open(out_path, 'w') as fp:
            #     for ii in range(num_frames):
            #         fp.write('%f\n'%depth_values[ii])
