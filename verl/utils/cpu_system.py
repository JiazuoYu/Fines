
import subprocess
import psutil

def get_system_stats():
    
    cpu_usage = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    mem_usage = 100 * (1 - mem.used / mem.total)
    # print(cpu_usage, mem_usage)
    return cpu_usage, mem_usage

if __name__ == "__main__":
    get_system_stats()