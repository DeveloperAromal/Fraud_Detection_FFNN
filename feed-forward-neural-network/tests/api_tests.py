import requests



def get_full_url(route, prefix):
    
    return f"http://127.0.0.1:8000{prefix}{route}"


def test_api(data, route, prefix="/api/v1/"):
    
    url = get_full_url(route, prefix)
    
    try:
        response = requests.post(url, json=data, timeout=10)
        
        return {
                  "status code": response.status_code,
                  "data": response.json()
               }
        
    except requests.ConnectionError:
        return {"status_code": None, "error": "ConnectionError: Could not connect to API"}
    
    except requests.Timeout:
        return {"status_code": None, "error": "Timeout: API request took too long"}
    
    except requests.RequestException as e:
        return {"status_code": None, "error": str(e)}
    
    except ValueError:
        return {"status_code": response.status_code, "error": "Invalid JSON response"}