from mmdet.models import Darknet
import torch

def main():
    # parse conf yaml

    # for networks
        # use netowrk get_graph to build network
        # use network extract to build extracted graph

        # use alalysize to extract some info

        # serilaze and save

    darknet = Darknet(depth=53)
    inputs = torch.rand(1, 3, 416, 416)
    level_outputs = darknet.forward(inputs)
    print(level_outputs)

if __name__ == '__main__':
    main()