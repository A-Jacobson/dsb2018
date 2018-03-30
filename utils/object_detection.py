import numpy as np
import torch


def extract_bboxes(masks):
    # from https://github.com/matterport/Mask_RCNN/blob/master/utils.py
    """Compute bounding boxes from masks.
    masks: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (x1, y1, x2, y2)].
    """
    masks = masks.numpy()
    boxes = np.zeros([masks.shape[0], 4], dtype=np.float32)
    for i in range(masks.shape[0]):
        m = masks[i, :, :]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No masks for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    # boxes = boxes[np.all(boxes, axis=1)]  # throw out zero boxes
    return torch.from_numpy(boxes)


def convert_wh(bbox):
    """
    converts an (x1, y1, x2, y2) bbox to an (x, y, width, height)
    """
    wh_bbox = np.zeros_like(bbox)
    wh_bbox[0] = bbox[0]
    wh_bbox[1] = bbox[1]
    wh_bbox[2] = bbox[2] - bbox[0]
    wh_bbox[3] = bbox[3] - bbox[1]
    return wh_bbox


class AnchorHelper:

    def __init__(self,
                 areas=(16, 32, 64, 128, 256),
                 ratios=(0.5, 1, 2),
                 scales=(2 ** 0, 2 ** (1. / 3.), 2 ** (2. / 3.)),
                 pyramid_levels=(3, 4, 5, 6, 7),
                 positive_overlap=0.5,
                 negative_overlap=0.3):
        self.areas = areas
        self.ratios = ratios
        self.scales = scales
        self.pyramid_levels = pyramid_levels
        self.pyramid_strides = [2 ** x for x in pyramid_levels]
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap

    def _top_left_anchors(self, area):
        """
        generates anchors (x1, y1, x2, y2)  for top left pixel in feature map
        """
        num_anchors = len(self.ratios) * len(self.scales)
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = area * np.tile(self.scales, (2, len(self.ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]  # W * H

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    def _anchors_for_featuremap(self, top_left_anchors, features_shape, stride):
        """
        generates all anchors for feature map by shifting top_left anchors over the feature map
        """
        shift_x = (np.arange(0, features_shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, features_shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(),
            shift_x.ravel(), shift_y.ravel()
            )).transpose()

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = top_left_anchors.shape[0]
        K = shifts.shape[0]
        anchors = (top_left_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
        return anchors.reshape((K * A, 4))

    def generate_anchors(self, image_size):
        """
        :param image_size: tuple (H , W) ex: (256, 256)
        """
        image_size = np.array(image_size)
        anchors = np.zeros((0, 4))
        for idx, p in enumerate(self.pyramid_levels):
            top_left_anchors = self._top_left_anchors(area=self.areas[idx])  # top left anchor for feature map
            anchors_one_level = self._anchors_for_featuremap(top_left_anchors,
                                                             image_size // self.pyramid_strides[idx],
                                                             self.pyramid_strides[idx])
            anchors = np.append(anchors, anchors_one_level, axis=0)
        return anchors

    def assign_gt_boxes(self, image, gt_boxes, class_labels=None):
        """
        assigns gt boxes to most overlapping anchor boxes, each gt_box can have multiple assigned anchors.
        """
        image_shape = (image.size(-1), image.size(-2))
        anchors = self.generate_anchors(image_shape)
        gt_boxes = np.array(gt_boxes)

        if class_labels is None:
            class_labels = np.ones(len(gt_boxes))
        # label: 1 is positive, 0 is negative, -1 is don't care
        labels = np.ones(len(anchors)) * -1

        # obtain indices of gt annotations with the greatest overlap
        overlaps = compute_overlap(anchors, gt_boxes)
        argmax_overlaps_inds = np.argmax(overlaps, axis=1)
        max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]

        # assign bg labels first so that positive labels can clobber them
        labels[max_overlaps < self.negative_overlap] = 0

        # compute box regression targets
        assigned_boxes = gt_boxes[argmax_overlaps_inds]
        class_labels = class_labels[argmax_overlaps_inds]  # expand class labels and targets

        # fg label: above threshold IOU
        positive_indices = max_overlaps >= self.positive_overlap
        labels[positive_indices] = 0
        labels[positive_indices] = class_labels[positive_indices]

        # ignore annotations outside of image
        anchors_centers = np.vstack([(anchors[:, 0] + anchors[:, 2]) / 2, (anchors[:, 1] + anchors[:, 3]) / 2]).T
        indices = np.logical_or(anchors_centers[:, 0] >= image_shape[1], anchors_centers[:, 1] >= image_shape[0])
        labels[indices] = -1
        return labels, assigned_boxes, anchors

    def make_targets(self, image, gt_boxes, class_labels=None):
        """
        Generates Anchors for image
        Assigns anchors to gt_boxes
        Computes bbox deltas for assigned anchors
        :return: retinanet class_labels and anchor_deltas
        """
        labels, assigned_boxes, anchors = self.assign_gt_boxes(image, gt_boxes, class_labels)
        anchor_deltas = get_deltas(anchors, assigned_boxes)
        return torch.LongTensor(labels), torch.FloatTensor(anchor_deltas)


def get_deltas(anchors, gt_boxes):
    """Compute bounding-box regression targets
    (log(difference) between anchors and gt_boxes) for an image."""

    # transform from (x1, y1, x2, y2) -> (x_ctr, y_ctr, w, h)
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    # clip widths to 1
    gt_widths = np.maximum(gt_widths, 1)
    gt_heights = np.maximum(gt_heights, 1)

    targets_dx = (gt_ctr_x - anchor_ctr_x) / anchor_widths
    targets_dy = (gt_ctr_y - anchor_ctr_y) / anchor_heights
    targets_dw = np.log(gt_widths / anchor_widths)
    targets_dh = np.log(gt_heights / anchor_heights)

    deltas = np.stack((targets_dx, targets_dy, targets_dw, targets_dh)).T
    return deltas


def apply_deltas(boxes, deltas):
    """
    apply predicted deltas to anchor (prior) boxes
    """
    boxes = boxes.reshape(-1, 4)
    deltas = deltas.reshape(-1, 4)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = np.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=1)
    pred_boxes = np.expand_dims(pred_boxes, axis=0)
    return pred_boxes


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum supression and returns indicies of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indicies into ixs[1:], so add 1 to get
        # indicies into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indicies of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)
