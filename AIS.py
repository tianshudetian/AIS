class AIScrawler:

    def __init__(self, MMSIs):
        self.data = MMSIs
        self.newdata = []

    def requestShipxy(self):
        import requests
        import pandas as pd
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'
        }
        session = requests.Session()
        main_url = 'http://www.shipxy.com/'  # 推测对该url发起请求会产生cookie
        session.get(main_url, headers=headers)
        url = 'http://www.shipxy.com/ship/GetShip'
        newData = []
        for MMSI in self.data:
            params = {
                'mmsi': MMSI,
            }
            page_text = session.get(url, headers=headers, params=params).json()
            try:
                tem = page_text['data'][0]
                ndata = [MMSI, tem['type'], tem['length']/10]
                newData.append(ndata)
            except:
                pass
        self.newdata = pd.DataFrame(newData, columns=['MMSI', 'Ship Type', 'Length'])






