
def main():
    # parse conf yaml

    # for networks
        # use netowrk get_graph to build network
        # use network extract to build extracted graph

        # use alalysize to extract some info

        # serilaze and save

    import paddle
    lenet = paddle.vision.models.LeNet(num_classes=10)
    paddle.summary(lenet,(1, 1, 28, 28))


if __name__ == '__main__':
    main()