from torchvision import models
from conv_summary import summary
from info import ConvInfo, Tail, AnalysizeRes, FlexGemmConvInfo
import pandas as pd
import math
from collections import OrderedDict
from models.get_model import get_model

MAX_BLOCK_NUM = 108 * 2

# FORWARD_TILES = [Tail(256, 64), Tail(256, 96), Tail(192, 64), Tail(384, 64)]
# BACKWARD_DATA_TILES = [Tail(256, 64), Tail(256, 96), Tail(192, 64), Tail(384, 64)]
# BACKWARD_WEIGHTS_TILES = [Tail(256, 64), Tail(256, 96), Tail(192, 64), Tail(384, 64)]

# FORWARD_TILES = [Tail(256, 64), Tail(256, 96), Tail(192, 64), Tail(384, 64), Tail(64, 64), Tail(128, 64), Tail(128, 96)]
FORWARD_TILES = [
    Tail(32, 64),
    Tail(32, 128),
    Tail(64, 64),
    Tail(64, 128),
    Tail(128, 32),
    Tail(128, 64),
    Tail(128, 80),
    Tail(192, 48),
    Tail(192, 64),
    Tail(256, 64),
    Tail(256, 96),
    Tail(256, 112),
    Tail(256, 128),
    Tail(384, 64),
    Tail(384, 80)
]

BACKWARD_DATA_TILES = FORWARD_TILES
BACKWARD_WEIGHTS_TILES = FORWARD_TILES

BATCH_SIZES = [64, 128]

INPUT_BATCH_SIZES = {
    "resnet50": [448],
    "resnet101": [112],
    "resnet152": [64],
    "se-resnext50": [448],
    "se-resnext101": [112],
    "se-resnext152": [64],
    "darknet-yolov3": [64],
    "maskrcnn": [20],
    "deeplabv3_drn_a_50": [32],
}

INPUT_SIZES = [(224, 224)]

# resnet50, resnet101, resnet152, se-resnext50, se-resnext101, se-resnext152, yolov3-darknet, maskrcnn

MODELS = OrderedDict(
    {
        "resnet50": models.resnet50(),
        "resnet101": models.resnet101(),
        "resnet152": models.resnet152(),
        "se-resnext50": get_model("se-resnext50"),
        "se-resnext101": get_model("se-resnext101"),
        "se-resnext152": get_model("se-resnext152"),
        "darknet-yolov3": get_model("darknet-yolov3"),
        "maskrcnn": models.resnet101(),
        "deeplabv3_drn_a_50": get_model("deeplabv3_drn_a_50"),
    }
)


def analysize(conv_info, tile):
    gemm_m = conv_info.batch_size * conv_info.output_h * conv_info.output_w
    output_channels = conv_info.output_channels

    tile_y = math.ceil(gemm_m / tile.ysize)
    tile_x = math.ceil(output_channels / tile.xsize)

    fake_usage = (tile_y * tile_x) / MAX_BLOCK_NUM
    ap_internal_usage = (gemm_m / tile.ysize) * (output_channels / tile.xsize) / (tile_y * tile_x)
    real_usage = fake_usage * ap_internal_usage

    ana_res = AnalysizeRes(tile_y, tile_x, fake_usage, real_usage, ap_internal_usage)
    return ana_res


def extract_conv_info(network, input_size):
    conv_summarys = summary(network, input_size)
    return conv_summarys


def get_tile_name(tile):
    return "x".join([str(tile.ysize), str(tile.xsize)])


def get_indexs(tiles):
    indexs = ["NETWORKNAME", "CONVNAME", "BATCHSIZE", "INPUT_CHANNEL", "INPUT_SIZE", "OUTPUT_CHANNEL", "KERNEL_SIZE",
              "STRIDE", "PADDING", "OUTPUT_SIZE"]
    for tile in tiles:
        name = get_tile_name(tile)
        indexs.append(name + '_TILE_NUMS')

    for tile in tiles:
        name = get_tile_name(tile)
        indexs.append(name + '_FAKE_AP_USAGE')
        # indexs.append(name + '_REAL_AP_USAGE')

    for tile in tiles:
        name = get_tile_name(tile)
        indexs.append(name + '_AP_INTERNAL_USAGE')

    df_dict = {}
    for index in indexs:
        df_dict[index] = []

    return df_dict


def init_score_dict(tiles):
    score_dict = {}

    for tile in tiles:
        name = get_tile_name(tile)
        score_dict[name] = 0

    return score_dict


def write_df_dict(network_name, df_dict, conv_info, origin_conv_info, tiles, score_dict):
    df_dict["NETWORKNAME"].append(network_name)
    for key, val in conv_info.get_dict_info().items():
        df_dict[key].append(val)

    best_tile = None
    best_score = -1
    for tile in tiles:
        ana_res = analysize(conv_info, tile)
        name = get_tile_name(tile)
        df_dict[name + '_TILE_NUMS'].append(f"({ana_res.tile_y}, {ana_res.tile_x})")
        df_dict[name + '_FAKE_AP_USAGE'].append(ana_res.fake_ap_usage)
        # df_dict[name + '_REAL_AP_USAGE'].append(ana_res.real_ap_usage)
        df_dict[name + '_AP_INTERNAL_USAGE'].append(ana_res.ap_internal_usage)

        if best_tile is None:
            best_tile = name
            best_score = ana_res.fake_ap_usage

        else:
            if math.modf(best_score) <= math.modf(ana_res.fake_ap_usage):
                best_tile = name
                best_score = ana_res.fake_ap_usage

    score_dict[best_tile] += 1


def main():
    df_forward_dict = get_indexs(FORWARD_TILES)
    df_backward_data_dict = get_indexs(BACKWARD_DATA_TILES)
    df_backward_weights_dict = get_indexs(BACKWARD_WEIGHTS_TILES)

    score_dict = init_score_dict(FORWARD_TILES)

    for nerwork_name, model in MODELS.items():
        print(f"network: {nerwork_name}")
        for input_size in INPUT_SIZES:
            for batch_size in INPUT_BATCH_SIZES[nerwork_name]:
                print(f"batch_size: {batch_size}")
                conv_infos = extract_conv_info(model, (batch_size, 3, input_size[0], input_size[1]))
                for _, conv_forward_info in conv_infos.items():
                    tmp_flexgemm_conv_info = FlexGemmConvInfo(conv_forward_info)
                    tmp_flexgemm_conv_info.gen_back_info()
                    conv_backward_data_info = tmp_flexgemm_conv_info.backward_data_info
                    conv_backward_weights_info = tmp_flexgemm_conv_info.backward_weights_info

                    write_df_dict(nerwork_name, df_forward_dict, conv_forward_info, conv_forward_info, FORWARD_TILES, score_dict)
                    write_df_dict(nerwork_name, df_backward_data_dict, conv_backward_data_info, conv_forward_info, BACKWARD_DATA_TILES,
                                  score_dict)
                    write_df_dict(nerwork_name, df_backward_weights_dict, conv_backward_weights_info, conv_forward_info,
                                  BACKWARD_WEIGHTS_TILES, score_dict)

    df_forward = pd.DataFrame.from_dict(df_forward_dict)
    # df_forward.to_csv("./resnet50_forward.csv")

    df_backward_data = pd.DataFrame.from_dict(df_backward_data_dict)
    # df_backward_data.to_csv("./resnet50_backward_data.csv")

    df_backward_weights = pd.DataFrame.from_dict(df_backward_weights_dict)
    # df_backward_weights.to_csv("./resnet50_backward_weights.csv")

    df = pd.concat([df_forward, df_backward_data, df_backward_weights])

    df.to_csv("./benchmark.csv")

    # print(score_dict)
    my_dict = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    print(my_dict)

    with open('output/tile_priority.csv', 'w') as f:
        for key, val in my_dict:
            f.write("%s,%s\n" % (key, val))


if __name__ == '__main__':
    main()
