class ConvInfo(object):
    def __init__(self,
                 name="",
                 batch_size=1,
                 input_channels=1,
                 input_h=224,
                 input_w=224,
                 output_channels=256,
                 kernel_h=3,
                 kernel_w=3,
                 output_h=224,
                 output_w=224,
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=0):
        self.name = name
        self.batch_size = batch_size
        self.input_channels = input_channels
        self.input_h = input_h
        self.input_w = input_w
        self.output_channels = output_channels
        self.kernel_h = kernel_h
        self.kernel_w = kernel_w
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_h = output_h
        self.output_w = output_w


    def get_nchw(self):
        return self.batch_size, self.input_channels, self.input_h, self.input_w

    def get_kcrs(self):
        return self.output_channels, self.input_channels, self.kernel_h, self.kernel_w

    def get_nkpq(self):
        return self.batch_size, self.output_channels, self.output_h, self.output_w

    def get_dict_info(self):
        N, C, H, W = self.get_nchw()
        K, _, R, S = self.get_kcrs()
        N, _, P, Q = self.get_nkpq()

        return {
            "CONVNAME": self.name,
            "BATCHSIZE": N,
            "INPUT_CHANNEL": C,
            "INPUT_SIZE": "x".join([str(H), str(W)]),
            "OUTPUT_CHANNEL": K,
            "KERNEL_SIZE": "x".join([str(R), str(S)]),
            "STRIDE": self.stride,
            "PADDING": self.padding,
            "OUTPUT_SIZE": "x".join([str(P), str(Q)]),
        }

class FlexGemmConvInfo(object):
    def __init__(self, forward_info):
        self.forward_info = forward_info
        self.backward_data_info = ConvInfo()
        self.backward_weights_info = ConvInfo()

    def gen_back_info(self):
        self.gen_backward_data_info()
        self.gen_backward_wights_info()

    def gen_backward_data_info(self):
        N, C, H, W = self.forward_info.get_nchw()
        K, _, R, S = self.forward_info.get_kcrs()
        _, _, P, Q = self.forward_info.get_nkpq()

        stride = self.forward_info.stride
        padding = self.forward_info.padding

        self.backward_data_info.name = self.forward_info.name + "_back_data"
        self.backward_data_info.batch_size = N
        self.backward_data_info.input_channels = K
        self.backward_data_info.input_h = H - 1 + R
        self.backward_data_info.input_w = W - 1 + S
        self.backward_data_info.output_channels = C
        self.backward_data_info.kernel_h = R
        self.backward_data_info.kernel_w = S
        self.backward_data_info.stride = (1, 1)
        # self.backward_data_info.padding = (
        #     (H - 1 + R - self.backward_data_info.input_h) // 2,
        #     (W - 1 + S - self.backward_data_info.input_w) // 2,
        # )
        self.backward_data_info.padding = (0, 0)
        self.backward_data_info.dilation = 0
        self.backward_data_info.output_h = H
        self.backward_data_info.output_w = W

    def gen_backward_wights_info(self):
        N, C, H, W = self.forward_info.get_nchw()
        K, _, R, S = self.forward_info.get_kcrs()
        _, _, P, Q = self.forward_info.get_nkpq()

        stride = self.forward_info.stride
        padding = self.forward_info.padding

        self.backward_weights_info.name = self.forward_info.name + "_back_weights"
        self.backward_weights_info.batch_size = C
        self.backward_weights_info.input_channels = N
        self.backward_weights_info.input_h = H
        self.backward_weights_info.input_w = W
        self.backward_weights_info.output_channels = K
        self.backward_weights_info.kernel_h = P
        self.backward_weights_info.kernel_w = Q
        self.backward_weights_info.stride = (1, 1)
        self.backward_weights_info.padding = padding
        self.backward_weights_info.dilation = (stride[0] - 1, stride[1] - 1)
        self.backward_weights_info.output_h = R
        self.backward_weights_info.output_w = S


class Tail(object):
    def __init__(self, ysize=256, xsize=64):
        self.ysize = ysize
        self.xsize = xsize


class AnalysizeRes(object):
    def __init__(self, tile_y, tile_x, fake_ap_usage, real_ap_usage, ap_internal_usage):
        self.tile_y = tile_y
        self.tile_x = tile_x
        self.fake_ap_usage = fake_ap_usage
        self.real_ap_usage = real_ap_usage
        self.ap_internal_usage = ap_internal_usage
