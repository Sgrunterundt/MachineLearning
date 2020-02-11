import torchnet
import sys

if len(sys.argv) < 2:
    print("Please provide path to save network")

torchnet.run(sys.argv[1])

