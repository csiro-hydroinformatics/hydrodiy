#!/usr/bin/env python

'''
---------------------------------------- 
 Generate html pages with 
 
 author : JL
 version : 2.0
 datae of first version : 2013-11-18
 comment : based on code 
	/ehpdata/akabir/wafari-proj/wafari/wafari_xv_simulations_before.py

---------------------------------------- 
'''

import datetime as dt
import os
import json
import re
from wafari.workflow.iterate_all.iterate_all import iter_catchments

#------------------------------------------------------------
# Script options
#------------------------------------------------------------

# Wafari project
PROJECT = '/ehpdata/wafari/dev/project' # Latest project
#PROJECT = '/data/nas01/jenkins/jobs/wafari_dm_project' # project used by Jenkins

# web sites that are included in the comparison
#websites =['http://www.bom.gov.au/water/ssf',
#    'http://wafaridev/data/jenkins/dm/seasonal',
#    'http://wafaridev/data/jenkins/dm/monthly']
#name = 'bjp_vs_dm'

websites =['http://wafaridev/data/jenkins/dm/monthly_2013083000',
            'http://wafaridev/data/jenkins/dm/monthly_2013110100',
            'http://wafaridev/data/jenkins/dm/monthly']
name = 'dm_monthly_progresses'

# Defines list of products to include
product_list = ['skillscores']
for m in range(1,13):
    product_list.append('pit%2.2d'%m)
    product_list.append('quantiles%2.2d'%m)

#------------------------------------------------------------
# Folders
#------------------------------------------------------------
now = dt.datetime.now().date().isoformat()
chOUT = '/ehpdata/jlerat/benchmark/%s_%s'%(now,name)
if not os.path.exists(chOUT):
    os.mkdir(chOUT)

#------------------------------------------------------------
# Initialise
#------------------------------------------------------------
config = json.load(file('%s/wafari/report_dm.json'%PROJECT))

# Selected IDs
#IDs_list = ('110003','120002','405209','116006B','116010A','116014A',
#        '412028','412066','412029','206014','410730','410734',
#        '419005','403205','OVENS_TOT','403213','401012','401203',
#        '401013','401009')
#IDs_list = ('110003','120002')
IDs_list = []
for drainage, basin, catchment, IDs, prod  in iter_catchments(config):
    IDs_list.append(IDs[0])


# Fucntion to generate html code
def generate_html(html_table):
    html = ['<html>','<head>','<title> Comparison of XV products</title>',
        '<b><p>Websites : %s</p>'%('  ~  '.join(websites)),
        '<p>Date : %s</p>'%now,
        '<p>Product : %s</p></b>'%product,
        '</head>\n','<body>\n','<table>\n']

    html.append('<tr>\n')
    html.append('<td></td>\n')
    for webs in websites:
        html.append('<td><p style="font-size:x-large"=><b>%s</b></p></td>\n'%webs)
    html.append('</tr>\n')

    for line in html_table:
        html.append('<tr>')
        html.append('<td>%s</td>\n'%line[0])
        for cell in line[1:]:
            html.append('<td><img src="%s"/></td>\n'%cell)
        html.append('</tr>\n')

    html.append('</table>')
    html.append('</body>')
    html.append('</html>')

    return html
# end of generate_html


#------------------------------------------------------------
# Main code
#------------------------------------------------------------

# Loop on products
for product in product_list:
    html_table = []

    # Finds the month number from product name
    month = '00'
    se = re.search('[\d]{2}$',product)
    if se:
        month = se.group(0)

    # Loop on ids
    count = 1
    for drainage, basin, catchment, IDs, prod  in iter_catchments(config):
        id = IDs[0]
    
        # Filter selected ids
        if any([u==id for u in IDs_list]):
            line = ['%d) %s (%s, %s)'%(count, catchment, id, basin)]
            for webs in websites:
                if product=='skillscores':
                    line.append('%s/%s/%s/xv/%s_XV_1.png'%(webs,drainage,basin,id))
                if product.startswith('pit'):
                    line.append('%s/%s/%s/xv/%s/%s_XV_6_%s.png'%(webs,drainage,basin,month,id,month))
                if product.startswith('quantiles'):
                    line.append('%s/%s/%s/xv/%s/%s_XV_4_%s.png'%(webs,drainage,basin,month,id,month))
    
            html_table.append(line)
            count += 1
    # end of loop on catchments    
    
    h1 = generate_html(html_table)
    with file('%s/%s.html'%(chOUT,product), 'w') as f1:
        f1.writelines(h1)
    
