import os
import argparse


def get_data(data_dir, data_name, url, data_origin_name):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists(data_name)):
        import urllib
        zippath = os.path.join(data_dir, data_origin_name)
        urllib.urlretrieve(url, zippath)
        # decompress
        os.system("bzip2 -d %r" % data_origin_name)
    os.chdir("..")
    print "Got Data!"


def estimate_density(data_name, feature_size):
    """sample 10 times of a size of 1000 for estimating the density of the sparse dataset"""
    import random
    P = 0.01
    if not os.path.exists(data_name):
        raise Exception("Data is not there!")
    density = []
    for _ in xrange(10):
        num_non_zero = 0
        num_sample = 0
        with open(data_name) as f:
            for line in f:
                if (random.random() < P):
                    num_non_zero += len(line.split(" ")) - 1
                    num_sample += 1
        density.append(num_non_zero * 1.0 / (feature_size * num_sample))
    print sum(density) * 100 / len(density)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description="Benchmark sparse operators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data', type=str, default=True)
    parser.add_argument('--get', type=bool, default=False)
    parser.add_argument('--estimate', type=bool, default=False)
    parser.add_argument('--feature_size', type=int, default=0)
    args = parser.parse_args()

    
    data_dir = os.path.join(os.getcwd(), 'data')
    data_name = os.path.join(data_dir, args.data)
    new_name = os.path.join(data_dir, args.data + ".mini")
    data_origin_name = args.data + ".bz2"
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/" + data_origin_name 
    
    if args.get:
	try:
            get_data(data_dir, data_name, url, data_origin_name)
        except:
            Exception("Get data failed.")
    elif args.estimate and args.feature_size > 0:
        try:
            estimate_density(data_name, args.feature_size)
        except:
            Exception("Density estimation failed")
