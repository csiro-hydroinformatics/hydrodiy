import re, json

import requests
import datetime

import numpy as np
import pandas as pd

class HttpFAOError(Exception):
        pass


fao_url = 'http://data.fao.org/developers/api/v1/en/resources'


class HyFAO():

    def __init__(self):

        self.current_url = ''

    def list_databases(self):

        url = '%s/databases.json' % fao_url
        
        req = requests.get(url)

        js = req.json()

        self.current_url = req.url

        db = pd.DataFrame([it for it in js['result']['list']['items']])
        
        return db


    def list_datasets(self, database):

        url = '%s/%s/datasets.json' % (fao_url, database)

        params = {'fields':
                    'mnemonic%2Clabel%40en%2Cdescription%40en%2Cviews'}

        req = requests.get(url, params = params)

        self.current_url = req.url

        js = req.json()

        ds = pd.DataFrame([it for it in js['result']['list']['items']])
        
        return ds


    def dataset_info(self, database, dataset):

        url = '%s/%s/%s' % (fao_url, 
                        database, dataset)

        self.current_url = url

        params = {'fields': 'mnemonic%2Clabel%40en'}

        req = requests.get('%s/dimensions.json?' % url, params = params)
        js = req.json()
        dims = pd.DataFrame([it for it in js['result']['list']['items']])
        
        req = requests.get('%s/members.json?' % url, params = params)
        js = req.json()
        membs = pd.DataFrame([it for it in js['result']['list']['items']])
        
        req = requests.get('%s/cnt/members.json?' % url, params = params)
        js = req.json()
        countries = pd.DataFrame([it for it in js['result']['list']['items']])
        
        return dims, membs, countries


    def data(self, database, dataset, country=None, year=None):

        url = '%s/%s/%s' % (fao_url, 
                        database, dataset)

        params = {'fields': 'mnemonic%2Clabel%40en'}

        req = requests.get('%s/dimensions.json?' % url0, params = params)
        js = req.json()
        dims = pd.DataFrame([it for it in js['result']['list']['items']])
        
        req = requests.get('%s/members.json?' % url0, params = params)
        js = req.json()
        membs = pd.DataFrame([it for it in js['result']['list']['items']])
        
        req = requests.get('%s/cnt/members.json?' % url0, params = params)
        js = req.json()
        countries = pd.DataFrame([it for it in js['result']['list']['items']])
        
        return dims, membs, countries


