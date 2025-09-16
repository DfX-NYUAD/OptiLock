from ec.impl.kgs_ops.muxlink_base import MuxLinkBase
from ec.impl.kgs_solution import KGSSolution
from muxlink.muxlink import MuxLink
from muxlink.original import *
from utils.bench_parser import BenchParser
import ast
# we will pass the target file path and kgss data
# and then we get the solution and run the attack

# encode the netlist and return the original kgss data list
def netlist_encode(netlist_str, circuit_name):
    muxlink = MuxLink(circuit_name)
    kgss = muxlink.encode(netlist_str)
    # print(dict(kgss.graph.nodes()))
    return kgss

# this function is used to set up all the environment for our result 
def muxlink_setup(target_path):
    # Load the benchmark
    netlist = BenchParser.instance().parse_file(target_path)
    netlist_str = netlist.to_string()
    circuit_name = target_path.split("/")[-1].split(".bench")[0].split("_")[0]
    kgss_ori = netlist_encode(netlist_str, circuit_name)
    return kgss_ori

# Muxlink base initilization
def muxlink_init(target_path, locked_file_path, h_hop, kgss_data_index, train_mark, epochs):
    mux_instance = MuxLinkBase.instance()
    mux_instance.load(target_path, locked_file_path, h_hop, kgss_data_index, train_mark, epochs)
    return mux_instance
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi_Mux")
    parser.add_argument("--target-path", type=str, help="target path")
    parser.add_argument("--h-hop", type=int, help="hop size")
    parser.add_argument("--kgss-data", type=str, help="kgss data")
    parser.add_argument("--train-mark", type=str, help="train mark")
    args = parser.parse_args()
    target_path = args.target_path
    h_hop = args.h_hop
    train_mark = args.train_mark
    if train_mark == "False":
        train_mark = False
    else:
        train_mark = True
    # load the kgss data
    kgss_data_temp = args.kgss_data
    kgss_data= ast.literal_eval(kgss_data_temp)
    # load the kgss_ori
    kgss_ori = muxlink_setup(target_path)
    kgss_ori.data = kgss_data
    kgss_sol = kgss_ori # get the solution
    # initialize the muxlink 
    mux_instance = muxlink_init(target_path, target_path, h_hop, kgss_data, train_mark, 100)
    # attack the solution
    acc, prec, kpa = mux_instance.attack(kgss_sol)
    




