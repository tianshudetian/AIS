class AISCrawler():
    def __init__(self, MMSIs):
        self.data = MMSIs
        cData = []

    def requestShipxy(self):
        '''
        equest Shipxy.com
        '''
        import requests
        url = "http://searchv3.shipxy.com/shipdata/search3.ashx"
        for MMSI in self.data:
            dic = {
                'f': 'srch',
                'kw': MMSI
            }
            rq = requests.get(url, params=dic, timeout=0.5)


