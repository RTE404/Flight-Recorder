from flightrec.clock import VectorClock, vc_rank, happens_before, concurrent


def test_tick_increments_own_component():
    vc = VectorClock("a")
    assert vc.tick() == {"a": 1}
    assert vc.tick() == {"a": 2}


def test_merge_takes_elementwise_max():
    vc = VectorClock("a", {"a": 1})
    vc.merge({"a": 0, "b": 3})
    assert vc.snapshot() == {"a": 1, "b": 3}
    vc.merge({"a": 5, "b": 1})
    assert vc.snapshot() == {"a": 5, "b": 3}


def test_merge_is_commutative_and_associative():
    vc1 = VectorClock("a")
    vc1.merge({"b": 2})
    vc1.merge({"c": 5})
    vc2 = VectorClock("a")
    vc2.merge({"c": 5})
    vc2.merge({"b": 2})
    assert vc1.snapshot() == vc2.snapshot()


def test_vc_rank_sums_components():
    assert vc_rank({"a": 2, "b": 3}) == 5
    assert vc_rank({}) == 0


def test_happens_before_true_when_strictly_dominated():
    assert happens_before({"a": 1}, {"a": 1, "b": 1})
    assert not happens_before({"a": 1, "b": 1}, {"a": 1})


def test_happens_before_false_for_equal_vectors():
    assert not happens_before({"a": 1}, {"a": 1})


def test_concurrent_true_for_incomparable_vectors():
    assert concurrent({"a": 1}, {"b": 1})
    assert not concurrent({"a": 1}, {"a": 1, "b": 1})
    assert not concurrent({"a": 1}, {"a": 1})
