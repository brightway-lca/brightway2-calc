from bw2calc.method_config import MethodConfig


def test_method_config_valid():
    data = {
        'impact_categories': [('foo', 'a'), ('foo', 'b')],
    }
    assert MethodConfig(**data)

    data = {
        'impact_categories': [('foo', 'a'), ('foo', 'b')],
        'normalizations': {('foo', 'a'): ('norm', 'standard'), ('foo', 'b'): ('norm', 'standard')}
    }
    assert MethodConfig(**data)

    data = {
        'impact_categories': [('foo', 'a'), ('foo', 'b')],
        'normalizations': {('foo', 'a'): ('norm', 'standard'), ('foo', 'b'): ('norm', 'standard')},
        'weightings': {('norm', 'standard'): ('weighting',)}
    }
    assert MethodConfig(**data)
