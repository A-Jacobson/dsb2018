import matplotlib.pyplot as plt

from .object_detection import convert_wh


def display_boxes(img, boxes):
    plt.imshow(img)
    for box in boxes:
        box = convert_wh(box)
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2], box[3],
                                          fill=False, lw=1, color='orange'))

# def show_preds(img, anchors, box_preds, class_preds):
#     box_preds = box_preds.data.numpy()
#     scores, indices = class_preds.max(dim=2)
#     scores = scores.data.numpy()[0]
#     indices = indices.data.numpy()[0]
#     preds = apply_anchor_deltas(anchors[np.where(indices > 0)], box_preds[0][np.where(indices > 0)])
#     keep = non_max_suppression(preds[0], scores[np.where(indices > 0)], threshold=0.4)
#     display_boxes(img.permute(1, 2, 0), preds[0][keep])
