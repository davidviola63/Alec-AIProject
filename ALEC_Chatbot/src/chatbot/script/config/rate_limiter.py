import time

class RateLimiter:
    def __init__(self, rpm: int, tpm: int, rpd: int):
        self.rpm = rpm; self.tpm = tpm; self.rpd = rpd
        self.win_min = []; self.win_day = []
        self.token_min = 0
        self.curr_min = int(time.time() // 60)
        self.curr_day = int(time.time() // 86400)

    def _roll(self):
        now_min = int(time.time() // 60)
        now_day = int(time.time() // 86400)
        if now_min != self.curr_min:
            self.curr_min = now_min; self.win_min = []; self.token_min = 0
        if now_day != self.curr_day:
            self.curr_day = now_day; self.win_day = []

    def check(self, est_tokens_out: int = 1000):
        self._roll()
        if len(self.win_min) >= self.rpm:
            raise RuntimeError("Rate limit RPM locale raggiunto.")
        if self.token_min + est_tokens_out > self.tpm:
            raise RuntimeError("Rate limit TPM locale raggiunto.")
        if len(self.win_day) >= self.rpd:
            raise RuntimeError("Rate limit RPD locale raggiunto.")

    def commit(self, est_tokens_out: int = 1000):
        self._roll()
        now = time.time()
        self.win_min.append(now); self.win_day.append(now)
        self.token_min += est_tokens_out
