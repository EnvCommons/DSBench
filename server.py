from openreward.environments import Server

from analysis import DSBenchAnalysis
from modeling import DSBenchModeling

if __name__ == "__main__":
    server = Server([DSBenchAnalysis, DSBenchModeling])
    server.run()
