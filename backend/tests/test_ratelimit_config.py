import pytest

from app.config import Settings
from app.ratelimit import RateLimiter, client_ip


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_allows_up_to_the_limit_then_blocks_with_retry_after():
    clock = FakeClock()
    rl = RateLimiter(3, window_s=60, clock=clock)
    assert [rl.check("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry = rl.check("a")
    assert allowed is False and retry == 60
    clock.t += 45
    assert rl.check("a") == (False, 15)                                 # counts down
    clock.t += 15.01
    assert rl.check("a")[0] is True                                     # window has passed


def test_keys_are_independent():
    rl = RateLimiter(1, clock=FakeClock())
    assert rl.check("a")[0] and rl.check("b")[0]
    assert not rl.check("a")[0] and not rl.check("b")[0]


def test_sliding_window_not_fixed_buckets():
    clock = FakeClock()
    rl = RateLimiter(2, window_s=10, clock=clock)
    rl.check("a"); clock.t += 6; rl.check("a"); clock.t += 5            # first hit is 11 s old, second 5 s old
    assert rl.check("a")[0] is True                                     # only one hit still in the window
    assert rl.check("a")[0] is False


def test_blocked_requests_do_not_extend_the_block():
    clock = FakeClock()
    rl = RateLimiter(1, window_s=10, clock=clock)
    rl.check("a")
    for _ in range(50):
        clock.t += 0.1
        assert rl.check("a")[0] is False                                # hammering while blocked
    clock.t += 10
    assert rl.check("a")[0] is True                                     # still released 10 s after the ONE counted hit


def test_memory_is_bounded():
    clock = FakeClock()
    rl = RateLimiter(5, window_s=60, max_keys=50, clock=clock)
    for i in range(500):
        rl.check(f"ip{i}")
        clock.t += 0.01
    assert len(rl) <= 51


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        RateLimiter(0)
    with pytest.raises(ValueError):
        RateLimiter(5, window_s=0)


def test_client_ip_ignores_forwarded_header_unless_trusted():
    h = {"x-forwarded-for": "203.0.113.9, 10.0.0.1"}
    assert client_ip(h, "10.0.0.5", trust_proxy=False) == "10.0.0.5"    # spoofable header ignored by default
    assert client_ip(h, "10.0.0.5", trust_proxy=True) == "203.0.113.9"
    assert client_ip({}, "10.0.0.5", trust_proxy=True) == "10.0.0.5"
    assert client_ip({}, None, trust_proxy=False) == "unknown"
    assert len(client_ip({"x-forwarded-for": "x" * 500}, None, True)) == 64


def test_settings_defaults_and_parsing(tmp_path):
    s = Settings.from_env({})
    assert s.cors_origins == ("http://localhost:3000",) and s.rate_match_per_min == 20 and not s.trust_proxy_headers
    s = Settings.from_env({"CORS_ORIGINS": "https://a.example, https://b.example", "RATE_MATCH_PER_MIN": "5",
                           "TRUST_PROXY_HEADERS": "TRUE", "DATA_DIR": str(tmp_path), "CORS_ORIGIN_REGEX": "https://x-.*"})
    assert s.cors_origins == ("https://a.example", "https://b.example") and s.rate_match_per_min == 5
    assert s.trust_proxy_headers and s.data_dir == tmp_path and s.cors_origin_regex == "https://x-.*"


def test_settings_reject_bad_values():
    with pytest.raises(ValueError, match="RATE_MATCH_PER_MIN"):
        Settings.from_env({"RATE_MATCH_PER_MIN": "lots"})
    with pytest.raises(ValueError, match="RATE_DEFAULT_PER_MIN"):
        Settings.from_env({"RATE_DEFAULT_PER_MIN": "0"})
    with pytest.raises(ValueError, match="explicit origins"):
        Settings.from_env({"CORS_ORIGINS": "*"})
