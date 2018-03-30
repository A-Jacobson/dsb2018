from utils.object_detection import AnchorHelper


def test_num_anchors():
    """
    Make sure we are generating the correct number of anchors
    """
    anchor_helper = AnchorHelper()
    pixels_all_feature_maps = (2 ** 2 + 4 ** 2 + 8 ** 2 + 16 ** 2 + 32 ** 2)
    anchors_per_pixel = 9
    anchors = anchor_helper.generate_anchors(image_size=(256, 256))
    assert anchors.shape == (pixels_all_feature_maps * anchors_per_pixel, 4)
