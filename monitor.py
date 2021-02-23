import psutil
import logging
#import GPUtil
from threading import Thread
import time
import tracemalloc
import prometheus_client
import atexit

class Monitor(Thread):
    def __init__(self, delay):
        super(Monitor, self).__init__()
        self.system_usage = prometheus_client.Gauge('system_usage',
                                       'Hold current system resource usage',
                                       ['resource_type'])
        prometheus_client.start_http_server(9999)
        self.started_request = False
        self.request_time = 0.0
        self.all_request_time = 0.0
        self.requests = 0
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        tracemalloc.start()
        self.start_time = 0
        self.stop_time = 0
        self.start()

    def run(self):
        while not self.stopped:
            #logging.info(GPUtil.showUtilization())
            self.system_usage.labels('CPU').set(psutil.cpu_percent())
            self.system_usage.labels('Memory').set(psutil.virtual_memory()[2])
            if self.started_request:
                request_average = self.all_request_time/self.requests
                self.system_usage.labels("Average Request Time").set(request_average)
                self.system_usage.labels("Latest Request Time").set(self.request_time)
            time.sleep(self.delay)

    def stop(self):
        current, peak = tracemalloc.get_traced_memory()
        logging.info(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()
        logging.info("Application took %f seconds to run", time.perf_counter() - self.start_time)
        self.stopped = True
    
    def exit_handler(self):
        self.stop()

    def start_request(self):
        self.start_time = time.perf_counter()

    def stop_request(self):
        self.stop_time = time.perf_counter() - self.start_time
        self.set_request_time(self.stop_time)

    def set_request_time(self, time):
        self.request_time = time
        self.all_request_time += time
        self.requests += 1
        self.started_request = True


if __name__ == "__main__":
    """testing purposes only"""
    mon = Monitor(2)
    time.sleep(1)
    mon.start_request()
    time.sleep(1)
    mon.stop_request()
    mon.start_request()
    time.sleep(2)
    mon.stop_request()