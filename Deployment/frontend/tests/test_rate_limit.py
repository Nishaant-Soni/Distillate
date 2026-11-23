"""The sliding windows: a limit per window, expiry, and a refused request not counted twice."""

from demo_app.rate_limit import SlidingWindow, check


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_the_limit_holds_within_a_window_and_frees_up_after_it():
    clock = Clock()
    window = SlidingWindow(2, window_s=60, clock=clock)
    assert window.acquire() == 0 and window.acquire() == 0
    clock.now += 10
    assert window.acquire() == 50          # the first event expires 50 s from now
    clock.now += 50
    assert window.acquire() == 0


def test_wait_records_nothing():
    window = SlidingWindow(1, clock=Clock())
    assert window.wait_s() == 0 and window.wait_s() == 0
    assert window.acquire() == 0
    assert window.wait_s() > 0


def test_a_session_over_its_limit_doesnt_use_up_the_instance():
    clock = Clock()
    session = SlidingWindow(1, clock=clock)
    instance = SlidingWindow(2, clock=clock)
    assert check(session, instance) == 0
    assert check(session, instance) > 0            # the session's limit
    other = SlidingWindow(1, clock=clock)
    assert check(other, instance) == 0             # the instance still had room for one more
    third = SlidingWindow(1, clock=clock)
    assert check(third, instance) > 0              # now the instance is full
    assert third.wait_s() == 0                     # and the refused session wasn't charged
