import requests
import numpy as np
import json
headers = {
		    'Content-Type': 'application/json',
	}
signal=np.random.rand(50000)
r = requests.post("http://127.0.0.1:5001/classification", headers=headers,data=json.dumps(signal.tolist()))
print(r.text)