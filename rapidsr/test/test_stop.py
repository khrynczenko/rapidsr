from rapidsr import stop


class TestStopWhenValidationLossHasNotImproved:
    def test_is_condition_satisfied_stops(self):
        stops = [stop.StopInfo(0, 0.5, i) for i in range(49)]
        stops.append(stop.StopInfo(0, 0.1, 50))
        condition = stop.StopWhenValidationLossHasNotImproved(50)
        results = [condition.is_satisfied(info) for info in stops]
        assert not all(results[:-1])
        assert results[-1] is True

    def test_is_condition_satisfied_doesnt_stop(self):
        stops = [stop.StopInfo(0, 0.5, i) for i in range(49)]
        stops.append(stop.StopInfo(0, 0.6, 50))
        condition = stop.StopWhenValidationLossHasNotImproved(50)
        results = [condition.is_satisfied(info) for info in stops]
        assert not all(results)
