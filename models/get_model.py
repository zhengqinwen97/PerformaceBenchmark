import torch.nn as nn
import torch


def get_model(network_name):
    input, model = None, None
    if network_name == "se-resnext50":
        from models.se_resnext.se_resnet import se_resnext_50
        model = se_resnext_50()
    elif network_name == "se-resnext101":
        from models.se_resnext.se_resnet import se_resnext_101
        model = se_resnext_101()
    elif network_name == "se-resnext152":
        from models.se_resnext.se_resnet import se_resnext_152
        model = se_resnext_152()
    elif network_name == "maskrcnn":
        # maskrcnn 的backbone其实就是resnet50
        pass
    elif network_name == "darknet-yolov3":
        from yolov3_models.yolo import Model
        model = Model(cfg="/home/qizheng/code/backbone_extract/models/yolov3/yolov3-master/yolov3_models/yolov3.yaml")
    elif network_name == "deeplabv3_drn_a_50":
        from modeling.backbone.drn import drn_a_50
        model = drn_a_50(BatchNorm=nn.BatchNorm2d, pretrained=False)
    elif network_name == "deeplabv3_drn_c_58":
        from modeling.backbone.drn import drn_c_58
        model = drn_c_58(BatchNorm=nn.BatchNorm2d, pretrained=False)
    elif network_name == "deeplabv3_drn_d_54":
        from modeling.backbone.drn import drn_d_54
        model = drn_d_54(BatchNorm=nn.BatchNorm2d, pretrained=False)
    elif network_name == "deeplabv3_mbv2":
        from modeling.backbone.mobilenet import MobileNetV2
        model = MobileNetV2(output_stride=16, BatchNorm=nn.BatchNorm2d)
    elif network_name == "deeplabv3_xception":
        from modeling.backbone.xception import AlignedXception
        model = AlignedXception(BatchNorm=nn.BatchNorm2d, pretrained=False, output_stride=16)
    elif network_name == "cycle_gan_generator":  # gan 比较特殊 后面再研究
        pass

    return model


def main():
    model = get_model("deeplabv3_xception")


if __name__ == '__main__':
    main()
