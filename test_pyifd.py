from tests.validate_algo import validate_algo


def test_adq1():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_ADQ1.mat', 'ADQ1') is True


def test_adq2():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_ADQ2.mat', 'ADQ2') is True


def test_adq3():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_ADQ3.mat', 'ADQ3') is True


def test_blk():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_BLK.mat', 'BLK') is True


def test_cagi():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_CAGI.mat', 'CAGI') is True


def test_ela():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_ELA.mat', 'ELA') is True


def test_gho():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_GHO.mat', 'GHO') is True


def test_noi1():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NOI1.mat', 'NOI1') is True


def test_noi2():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NOI2.mat', 'NOI2') is True


def test_noi4():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NOI4.mat', 'NOI4') is True


def test_noi5():
    assert validate_algo('tests/data/168_image.jpg', 'tests/data/168_NOI5.mat', 'NOI5') is True
