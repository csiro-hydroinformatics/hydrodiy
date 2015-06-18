import re, json

import requests
import datetime

import numpy as np
import pandas as pd

class HttpFAOError(Exception):
        pass


fao_url = ('http://data.fao.org/developers/'
                'api/v1/en/resources')


class HyFAO():

    def __init__(self):

        self.current_url = ''

    def databases(self):
        ''' Returns a pandas data frame containing the list of FAO databases '''

        url = '%s/databases.json' % fao_url
        
        req = requests.get(url)

        js = req.json()

        self.current_url = req.url

        db = pd.DataFrame([it 
            for it in js['result']['list']['items']])

        db['mnemonic'] = db['urn'].apply(lambda x:
                                re.sub('.*\\:', '', x))
        
        return db


    def datasets(self, database):
        ''' Returns a pandas data frame containing the list of datasets in a given FAO database '''

        url = '%s/%s/datasets.json' % (fao_url, database)

        params = {'fields':
                    'mnemonic%2Clabel%40en%2C'
                    'description%40en%2Cviews'}

        req = requests.get(url, params = params)

        self.current_url = req.url

        js = req.json()

        ds = None 

        if 'items' in js['result']['list']:

            ds = pd.DataFrame([it 
                for it in js['result']['list']['items']])
        
        return ds

    def countries(self):
        ''' Returns a pandas data frame with the list of countries and their iso 3 code '''

        url = ('http://data.fao.org/statistics/named-query?'
                'database=countryprofiles&'
                'queryName=country-list&'
                'authKey=d30aebf0-ab2a-'
                '11e1-afa6-0800200c9a66&version=1.0') 

        req = requests.get(url)

        js = req.json()

        countries = pd.DataFrame([it 
                for it in js['result']['list']['items']])

        return countries



    def dataset_info(self, database, dataset):
        ''' Returns 3 pandas data frame containing the dimensions, members and countries in a given FAO dataset '''

        url = '%s/%s/%s' % (fao_url, 
                        database, dataset)

        self.current_url = url

        params = {'fields': 'mnemonic%2Clabel%40en'}

        req = requests.get('%s/dimensions.json?' % url, 
                params = params)

        js = req.json()
        dims = pd.DataFrame([it 
            for it in js['result']['list']['items']])
        
        req = requests.get('%s/members.json?' % url, 
                params = params)

        js = req.json()
        membs = pd.DataFrame([it 
            for it in js['result']['list']['items']])
        
        req = requests.get('%s/cnt/members.json?' % url, 
                params = params)

        js = req.json()
        countries = pd.DataFrame([it 
            for it in js['result']['list']['items']])
        
        return dims, membs, countries


    def data(self, database, dataset, 
            country=None, year=None):

        return data

