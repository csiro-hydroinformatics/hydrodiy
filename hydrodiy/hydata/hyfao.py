import re, json

import requests
import datetime

import numpy as np
import pandas as pd

class HttpFAOError(Exception):
        pass

faostat_urls = ['http://faostat3.fao.org/faostat-api/rest',
        'http://faostat3.fao.org/wds/api']

object_columns = {
        'countries':['country_code', 'country_label'], 
        'regions':['region_code', 'region_label'], 
        'specialgroups':['specialgroup_code', 
            'specialgroup_label'], 
        'items':['item_code', 'item_label'],
        'elements':['element_code', 'element_label'],
        'itemsaggregated':['itemagg_code', 'itemagg_label']
    }


class HyFAO():

    def __init__(self):

        self.current_url = ''

    def get_domain_codes(self):

        # Run query
        base_url = ('%s/groupsanddomains/faostat/E') % (
                        faostat_urls[0])

        req = requests.get(base_url)

        js = req.json()

        # Convert json object to dataframe
        df = pd.DataFrame(js)
        
        df.columns = ['group_code', 'group_label',
                    'domain_code', 'domain_label',
                    'order']

        return df


    def get_object_codes(self, object_name, domain_code):

        if not object_name in object_columns:
            raise ValueError(('object_name(%s) not in ['
                '%s]') % (object_name, 
                    '/'.join(object_columns.keys())))

        # Run query
        base_url = ('%s/procedures/%s/faostat/%s/E') % (
                        faostat_urls[0], 
                        object_name, domain_code)

        req = requests.get(base_url)

        self.current_url = req.url

        # Convert json object to dataframe
        js = req.json()

        df = pd.DataFrame(js)

        cc = object_columns[object_name]

        df.columns = cc + ['X%d' % (i+1) 
                            for i in range(df.shape[1]-len(cc))]

        df['object_name'] = object_name
        df['domain_code'] = domain_code

        # Remove useless columns
        cc = [cn for cn in df.columns if re.search('^X', cn)]
        df = df.drop(cc, axis=1)

        return df

    def search_object_codes(self, object_name, domain_code, 
                pattern):

        object_codes = self.get_object_codes(object_name, 
                                    domain_code)

        cc = [cn for cn in object_codes.columns 
                    if re.search('_label', cn)]
        
        labels = object_codes[cc].squeeze()

        idx = [bool(re.search(pattern, x)) for x in labels]

        return object_codes.loc[idx,:]


    def get_data(self, 
            domain_code,
            item_codes,
            element_codes,
            area_codes,
            years = range(2010, 2014)):

        ## Query parameters
        #params = {
        #    'db':'faostat',
        #    'select':'A.AreaCode[FAOST_CODE],'
        #                'D.year[Year],D.value[Value],'
        #                'A.AreaNameE[AreaName],'
        #                'E.elementnamee[ElementName],'
        #                'I.itemnamee[ItemName]',
        #    'from':'data[D],element[E],item[I],area[A]',
        #    'where': ('D.elementcode(%s),D.itemcode(%s),'
        #            'D.domaincode(\'%s\'),'
        #            'JOIN(D.elementcode:E.elementcode),'
        #            'JOIN(D.itemcode:I.itemcode),'
        #            'JOIN(D.areacode:A.areacode)') % (
        #                    element_code, item_code, domain_code),
        #    'orderby':'E.elementnamee,D.year',
        #    'out':'json'
        #}

        #if not countries is None:
        #    params['where'] = '%s,A.AreaCode(%s)' % (params['where'], 
        #                            ':'.join(countries))

        payload = {
                'datasource': 'faostat',
                'domainCode': domain_code,
                'lang' : 'E',
                'areaCodes': area_codes,
                'itemCodes' : item_codes,
                'elementListCodes' : element_codes,
                'years' : years,
                'flags': True,
                'codes': True,
                'units': True,
                'nullValues': False,
                'thousandSeparator' : ',',
                'decimalSeparator': '.',
                'decimalPlaces': 2,
                'limit':-1
        }

        # Request data
        #headers = {'Content-type': 'application/json',
        #            'Accept': 'text/plain'} 

        url = '%s/procedures/data' % faostat_urls[0]

        req = requests.post(url, json = payload)
                #data = json.dumps(payload),
                #headers = headers)

        print(req.status_code)
        import pdb; pdb.set_trace()

        if req.status_code != 200:
            raise HttpFAOError('URL : %s\nRequest status : %d' % 
                    (req.url, req.status_code))

        self.current_url = req.url

        # Convert json object to dataframe
        js = req.json()

        df = pd.DataFrame(js)

        df.columns = list(df.iloc[0, :].values)
        df = df.iloc[1:,:]

        df['domain_code'] = domain_code
        df['element_code'] = element_code
        df['item_code'] = item_code

        return df
    
