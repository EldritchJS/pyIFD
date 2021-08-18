from tests.validate_algo import validate_algo


def test_adq1():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_ADQ1.mat', 'ADQ1', 0.9) is True


def test_adq2():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_ADQ2.mat', 'ADQ2', 0.9) is True


def test_adq3():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_ADQ3.mat', 'ADQ3', 0.9) is True


def test_blk():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_BLK.mat', 'BLK', 0.85) is True


def test_cagi():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_CAGI.mat', 'CAGI', 0.9) is True


def test_cfa1():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_CFA1.mat', 'CFA1', 0.9) is True


def test_cfa2():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_CFA2.mat', 'CFA2', 0.9) is True


def test_dct():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_DCT.mat', 'DCT', 0.9) is True


def test_ela():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_ELA.mat', 'ELA', 0.9) is True


def test_gho():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_GHO.mat', 'GHO', 0.9) is True


def test_nadq():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NADQ.mat', 'NADQ', 0.9) is True


def test_noi1():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NOI1.mat', 'NOI1', 0.9) is True


def test_noi2():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NOI2.mat', 'NOI2', 0.9) is True


def test_noi4():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NOI4.mat', 'NOI4', 0.9) is True


def test_noi5():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NOI5.mat', 'NOI5', 0.85) is True
