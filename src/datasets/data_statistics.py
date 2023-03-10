def get_data_mean_and_stdev(dataset):
    if dataset == 'resisc45':
        mean = [0.377, 0.391, 0.363]
        std  = [0.203, 0.189, 0.186]
    elif dataset == 'eurosat':
        mean = [1354.3003, 1117.7579, 1042.2800,  947.6443, 1199.6334, 2001.9829, 2372.5579, 2299.6663,  731.0175,   12.0956, 1822.4083, 1119.5759, 2598.4456]
        std = [244.0469, 323.4128, 385.0928, 584.1638, 566.0543, 858.5753, 1083.6704, 1103.0342, 402.9594, 4.7207, 1002.4071, 759.6080, 1228.4104]
    elif dataset == 'so2sat_sen1':
        mean = [-3.6247e-05, -7.5790e-06,  6.0370e-05,  2.5129e-05,  4.4201e-02, 2.5761e-01,  7.5741e-04,  1.3503e-03]
        std = [0.1756, 0.1756, 0.4600, 0.4560, 2.8554, 8.3233, 2.4494, 1.4644]
    elif dataset == 'so2sat_sen2':
        mean = [0.1238, 0.1093, 0.1011, 0.1142, 0.1593, 0.1815, 0.1746, 0.1950, 0.1543, 0.1091]
        std = [0.0396, 0.0478, 0.0664, 0.0636, 0.0774, 0.0910, 0.0922, 0.1016, 0.0999, 0.0878]
    elif dataset == 'bigearthnet':
        mean = [1562.0203, 1561.3035, 1562.4702, 1559.7567, 1560.7955, 1564.3341, 1558.2031, 1560.1460, 1563.4475, 1563.5408, 1559.8683, 1562.0842]
        std = [1635.5913, 1633.8538, 1634.5411, 1634.0165, 1636.4137, 1635.9663, 1634.6973, 1633.7421, 1634.8866, 1638.0367, 1635.1881, 1636.3383]
    else:
        raise Exception(f'Dataset {dataset} not supported.')

    return mean, std
