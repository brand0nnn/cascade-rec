from cascade.analysis.switch_cases_devices import classify, generation, parse_title


def keys(title):
    return set(parse_title(title)[0])


def test_iphone_variants_and_lists():
    assert keys('Case for iPhone 13 Pro Max 6.7 inch') == {'iphone|13 pro max'}
    assert keys('Screen Protector for iPhone 11/XR, 6.1 Inch 3 Pack') == {'iphone|11', 'iphone|xr'}
    assert keys('Charger for iPhone 13/12/11/XS Max/X/8/7/6s Plus') == {
        'iphone|13', 'iphone|12', 'iphone|11', 'iphone|xs max', 'iphone|x', 'iphone|8', 'iphone|7', 'iphone|6s plus'}
    assert keys('Case for iPhone SE 2022 / iPhone SE 2020 / 8 / 7') == {'iphone|se2022', 'iphone|se2020', 'iphone|8', 'iphone|7'}
    assert keys('iPhone 12 mini 5G case 2 Pack') == {'iphone|12 mini'}
    assert keys('Lightning cable for iPhone, iPad') == set()
    assert parse_title('Lightning cable for iPhone, iPad')[1] == {'iphone'}


def test_galaxy_series():
    assert keys('Samsung Galaxy S21 Ultra 5G case') == {'galaxy_s|s21 ultra'}
    assert keys('Galaxy S10e holster') == {'galaxy_s|s10e'}
    assert keys('for Galaxy S22/S21/S20, Note 10 9') == {'galaxy_s|s22', 'galaxy_s|s21', 'galaxy_s|s20',
                                                         'galaxy_note|note10', 'galaxy_note|note9'}
    assert keys('Samsung Galaxy Z Flip 3 case') == {'galaxy_z_flip|flip3'}
    assert keys('Galaxy A52 5G screen protector') == {'galaxy_a|a52'}
    assert keys('Samsung Galaxy Watch 4 band') == set()
    assert keys('Samsung 2 Pack charger') == set()


def test_pixel_watch_fitbit():
    assert keys('Google Pixel 6 Pro screen protector') == {'pixel|6 pro'}
    assert keys('Pixel 4a 5G case') == {'pixel|4a'}
    assert keys('Google Pixel XL Case') == {'pixel|1 xl'}
    assert keys('Pixel 3 XL / 3XL glass') == {'pixel|3 xl'}
    assert keys('Google Pixel Buds case') == set()
    assert keys('Band Compatible with Apple Watch 38mm 40mm 44mm for iWatch Series 8/7/6/SE/3') == {
        'apple_watch|series 8', 'apple_watch|series 7', 'apple_watch|series 6', 'apple_watch|se', 'apple_watch|series 3'}
    assert keys('Apple Watch Ultra 49mm case') == {'apple_watch|ultra'}
    assert keys('Charger for Fitbit Versa 2 and Charge 4/3') == {'fitbit|versa 2', 'fitbit|charge 4', 'fitbit|charge 3'}


def test_generation_and_classify():
    assert generation('iphone|13 pro')[1] > generation('iphone|xr')[1]
    assert generation('galaxy_s|s20')[1] > generation('galaxy_s|s9')[1]
    assert generation('galaxy_a|a52') is None and generation('iphone|se') is None
    T = frozenset({'iphone|13'})
    c = classify(T, {'iphone|xr'}, {'iphone'}, {'iphone': 10.5}, frozenset({'iphone|xr'}))
    assert (c['cat'], c['generation'], c['disjoint_from_last']) == ('same_family_other_model', 'newer', True)
    assert classify(T, {'iphone|13'}, {'iphone'}, {'iphone': 13.0}, frozenset())['cat'] == 'model_in_history'
    assert classify(frozenset({'apple_watch|series 7'}), {'iphone|12'}, {'iphone'}, {'iphone': 12.0}, frozenset())['cat'] == 'new_family_same_company'
    assert classify(T, set(), set(), {}, frozenset())['cat'] == 'new_family_history_names_no_device'
    assert classify(T, {'pixel|3'}, {'pixel'}, {'pixel': 3.0}, frozenset())['cat'] == 'new_family_other_company'
