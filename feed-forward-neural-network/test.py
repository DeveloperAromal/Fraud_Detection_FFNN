from tests.api_tests import test_api


def test():
    
    data = {
        "features": [74.18, 30.0, 1989.0, 1.0, 22845.94, 5.94, 151.14, 16.0, 23.0, 4.0,
                    31.0, 17.0, 24.0, 16.0, 31.0, 1416.0, 6.71, 0.9957, 45.13, 0.14,
                    45.0, 46.06, 13.75, 5317.99, 0.000105, 1544.71]
    }

    result = test_api(data, route="fraud")
    print(result)
    
    
test()