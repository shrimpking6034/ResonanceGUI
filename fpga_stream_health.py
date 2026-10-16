"""Host-side delivery diagnostics; not a hardware sampling clock measurement."""
from collections import deque
import numpy as np


class DeliveryMonitor:
    def __init__(self):
        self.packets=deque(maxlen=4096)
        self.missing_packets=0
        self.samples=0

    def observe(self,batch):
        self.samples+=len(batch.values)
        self.missing_packets+=batch.missing_packets
        self.packets.append((batch.host_monotonic_s,self.samples,len(batch.values)))
        while len(self.packets)>4 and batch.host_monotonic_s-self.packets[0][0]>5:
            self.packets.popleft()

    def summary(self,expected_rate,local_dropped=0):
        rate=float("nan")
        if len(self.packets)>=4:
            first,last=self.packets[0],self.packets[-1]
            dt=last[0]-first[0]
            if dt>=.5:
                rate=(last[1]-first[1])/dt
        warnings=[]
        if local_dropped or self.missing_packets:
            warnings.append("데이터 누락: 시간축·분석 검토 필요")
        if np.isfinite(rate) and abs(rate/expected_rate-1)>.10:
            warnings.append("수신 속도와 CIC 예상값 불일치")
        return dict(host_delivery_rate_hz=rate,expected_rate_hz=expected_rate,
                    missing_packets_verified=self.missing_packets,
                    local_dropped_batches=local_dropped,warnings=warnings,
                    hardware_timestamps_available=False)
