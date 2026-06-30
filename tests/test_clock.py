from flightrec.clock import LamportClock


def test_tick_increments_and_returns():
    c = LamportClock()
    assert c.tick() == 1
    assert c.tick() == 2
    assert c.value == 2


def test_update_takes_max_plus_one():
    c = LamportClock()
    c.tick()  # value = 1
    assert c.update(5) == 6
    assert c.update(2) == 7
