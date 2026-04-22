from ..codeop import segment_ends_to_binary


def test_segment_ends_to_binary():
    result = segment_ends_to_binary([0, 1, 3, 6])
    expected = [1, 1, 0, 1, 0, 0, 1]
    assert result == expected
