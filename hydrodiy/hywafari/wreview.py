#!/usr/bin/env python

import os, sys, re
from datetime import datetime
from string import Template
from dateutil.relativedelta import relativedelta

import logging
from optparse import OptionParser

import json
import urllib2

import numpy as np
import pandas as pd

class wreview:
    
    def __init__(self, url, forcdate, test):

        self.logger = logging.getLogger('wreview')

        self.test = test
        
        self.url = url
        is_err, errmessage = self.__checkurl(url)
        if is_err:
            message = 'CAN\'T ACCESS FORECAST URL %s ' % url
            self.logger.error(message)
            raise ValueError(message)

        self.logger.info('FORECAST URL : %s' % url)

        self.forcdate = forcdate
        self.logger.info('FORECAST MONTH : %s' % forcdate)

        fs = '%s/website_config.json' % url
        is_err, errmessage = self.__checkurl(fs)
        if is_err:
            message = 'CAN\'T ACCESS FORECAT CONFIG %s ' % fs
            self.logger.error(message)
            raise ValueError(message)

        config = json.load(urllib2.urlopen(fs))
        self.config = config
       
        # Get list of sites
        sites = []
        for d in config['drainages']:
            for b in config['drainages'][d]['basins']:
                for c in config['drainages'][d]['basins'][b]['catchments']:
                    sites.append({'id':c[0], 'name':c[1], 'basin':b, 'drainage':d})
        sites = pd.DataFrame(sites)
        self.sites = sites
        self.logger.info('NB OF SITES IN CONFIG : %d' % sites.shape[0])
       
        # List of expected products for each site
        ep = {'FC tercile': '${base_url}/fc/${year}/${month}/${id}_FC_6_${year}_${month}.png',
            'FC hprob': '${base_url}/fc/${year}/${month}/${id}_FC_5_${year}_${month}.png',
            'FC hexceed': '${base_url}/fc/${year}/${month}/${id}_FC_4_${year}_${month}.png',
            'FC exceed': '${base_url}/fc/${year}/${month}/${id}_FC_1_${year}_${month}.png',
            'FC prob': '${base_url}/fc/${year}/${month}/${id}_FC_3_${year}_${month}.png',
            'XV skillscores summary': '${base_url}/xv/${id}_XV_1.png',
            'XV forecast quantile vs median': '${base_url}/xv/${month}/${id}_XV_2_${month}.png',
            'XV forecast quantile vs year': '${base_url}/xv/${month}/${id}_XV_4_${month}.png'}
        self.expected_products = ep

    def __checkurl(self, url):

        is_err = False
        errmessage = 'All good mate'
        try:
            urllib2.urlopen(url)

        except (urllib2.HTTPError, urllib2.URLError) as  err:
            is_err = True
            errmessage = str(err)

        return is_err, errmessage

    def inspectsite(self, id):

        sites = self.sites

        idx = sites['id'] == id
        basin = sites['basin'][idx].squeeze()
        drainage = sites['drainage'][idx].squeeze()

        self.logger.info('INSPECTING FORECASTS FOR SITE %s (%s, %s)' % (id, drainage, basin))

        # Base URL for specific site
        base_url = '%s/%s/%s' % (self.url, drainage, basin)

        # Loop through products and check urls
        ep = self.expected_products
        for p in ep:

            # Define start and end of product availability
            end = self.forcdate
            start = end

            if p.startswith('FC'):
                start = datetime(end.year-4, 1, 1)

            if p.startswith('XV forecast quantile'):
                start = end - relativedelta(years=1) + relativedelta(months=1)

            dates = pd.date_range(start, end, freq = 'MS')

            # Loop through dates and checl url of products
            for date in dates:
                tmp = Template(ep[p])

                # Check url 
                product_url = tmp.substitute(base_url=base_url, id=id, 
                    month='%2.2d' % date.month, year=date.year)

                is_error, errmessage = self.__checkurl(product_url)

                if is_error:
                    self.logger.error('%s for site %s on %s, %s' % (p, id, date, errmessage))
                    self.error_count += 1

                meta = {'id':id, 'basin':basin, 'drainage':drainage, 
                    'display_date':date.date(), 'type':p, 'url':product_url}
                meta['error'] = is_error
                meta['errmessage'] = errmessage
                self.product_list.append(meta)
                self.product_count += 1

    def inspectall(self):
 
        self.error_count = 0
        self.product_count = 0
        self.product_list = []
       
        self.logger.info('\n') 
        self.logger.info('\n') 
        self.logger.info('START REVIEW')
        self.logger.info('\n') 

        # Reduce site list for test runs
        sites = self.sites
        if self.test:
            sites = sites[:2]

        for idx, row in sites.iterrows():
            self.inspectsite(row['id'])

        pl = pd.DataFrame(self.product_list)
        self.product_list = pl

        self.product_stats = pd.pivot_table(pl, index=['drainage', 'basin', 'id'],
                columns = 'type', values = 'url', aggfunc=len)

        self.logger.info('\n') 
        self.logger.info('\n') 
        self.logger.info('NUMBER OF PRODUCTS REVIEWED: %d' % self.product_count)
        self.logger.info('NUMBER OF ERRORS: %d' % self.error_count)
        self.logger.info('REVIEW COMPLETED, SEE YA!')

if __name__ == "__main__":

    usage = '%prog [options]'
    parser = OptionParser(usage=usage)

    parser.add_option("-u", "--url", dest="url",
                  help="inspect URL", metavar="URL")

    parser.add_option("-f", "--filename", dest="filename",
                  help="write report to FILE", metavar="FILE")

    parser.add_option("-d", "--forcdate", dest="forcdate",
                  help="Assume forecast is produced for month DATE", metavar="DATE")

    parser.add_option("-t", "--test", action='store_true', dest="test", default = False,
                  help="Run in test mode (only 2 sites tested)")

    (options, args) = parser.parse_args()

    # Parse options
    if options.url is None:
        raise ValueError('Url is not defined')

    if options.forcdate is None:
        now = datetime.now()
        forcdate = datetime(now.year, now.month, 1)
    else:
        forcdate = datetime.strptime(options.forcdate, '%Y-%m-%d')

    # Logger options
    logger = logging.getLogger('wreview')
    logger.setLevel(logging.DEBUG)    

    ft = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s : %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(ft)
    logger.addHandler(ch)

    flog = options.filename
    if not flog is None:
        os.system('rm -f %s' % flog)    
        fh = logging.FileHandler(flog)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(ft)
        logger.addHandler(fh)

    # Run review
    rev = wreview(options.url, forcdate, options.test)
    rev.inspectall()

    if not flog is None:
        prod = rev.product_list
        flist = re.sub('\\.[^\\.]*$', '_prodlist.csv', flog)
        prod.to_csv(flist)

        prodstats = rev.product_stats
        fstats = re.sub('\\.[^\\.]*$', '_prodstats.csv', flog)
        prodstats.to_csv(fstats)


