from fcos.utils import Detections


class BuildTargets(object):
    def __init__(self, box_coder):
        self.box_coder = box_coder

    def __call__(self, input):
        dets = Detections(class_ids=input["class_ids"], boxes=input["boxes"], scores=None)

        _, h, w = input["image"].size()
        class_maps, loc_maps, cent_maps = self.box_coder.encode(dets, (h, w))

        return input["image"], (class_maps, loc_maps, cent_maps), dets
