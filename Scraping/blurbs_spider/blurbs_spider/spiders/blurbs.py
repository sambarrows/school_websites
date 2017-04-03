# -*- coding: utf-8 -*-
from scrapy import Spider, Request
from pandas import read_csv, DataFrame
import unicodedata
from datetime import datetime

from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError
from twisted.internet.error import TimeoutError


FILENAME = '/home/ec2-user/school_scraping/Data/csvurls.csv' 
# LOCAL_FILENAME = '/Users/sambarrows/Dropbox/School_websites/Scraping/csvurls.csv'

def make_clean_string(my_list, goop):
	clean_list = [element for element in my_list if element not in goop]
	clean_string = ' '.join(clean_list)
	if type(clean_string)=="unicode":
		clean_string = unicodedata.normalize("NFKD", clean_string)
	clean_string = clean_string.replace('\n', ' ').replace('\r', ' ')
	clean_string = ' '.join(clean_string.split())
	return clean_string

class BlurbsSpider(Spider):
	name = "blurbs"
	# allowed_domains = ['http://www.abbeyfederation.co.uk/']
	# start_urls = (
	# 	read_csv(FILENAME, names=['urn','url'])['url'].tolist()
	# )

	def start_requests(self):
		entries = read_csv(FILENAME, names=['urn','url'])
		for i in range(len(entries)):
			url = entries.loc[i,]['url']
			urn = entries.loc[i,]['urn']

			yield Request(url, callback=self.parse, errback=self.parse_error, dont_filter=True, meta={'urn': urn})

		# for u in self.start_urls:
		# 	yield Request(u, callback=self.parse, errback=self.parse_error, dont_filter=True)

	def parse_error(self, failure):

		if failure.check(HttpError):
			response = failure.value.response
			flag = 'HTTP_error'
		elif failure.check(DNSLookupError):
			response = failure.request
			flag = 'DNS_error'
		elif failure.check(TimeoutError):
			response = failure.request
			flag = 'Timeout_error'
		else:
			flag = 'Other_error'
			response = failure.request
		
		fin_blurb = ''
		used_url = response.url
		if hasattr(response, 'request'):
			if 'redirect_urls' in response.request.meta:
				orig_url = response.request.meta['redirect_urls'][0]
			else:
				orig_url = response.url
		else:
			if 'redirect_urls' in response.meta:
				orig_url = response.meta['redirect_urls'][0]
			else:
				orig_url = response.url
		if hasattr(response, 'request'):
			urn = response.request.meta['urn']
		else:
			urn = response.meta['urn']

		result = dict(url=used_url, blurb=fin_blurb, length=len(fin_blurb), time=str(datetime.now()), 
			urn=urn, orig_url=orig_url, flag=flag)
		yield result

	def parse(self, response):
		used_url = response.url
		if 'redirect_urls' in response.request.meta:
			orig_url = response.request.meta['redirect_urls'][0]
		else:
			orig_url = response.url
		urn = response.request.meta['urn']

		# Select tabs with "Welcome" in text and their following siblings, grouping by each Welcome and joining
		# each group into a single string. Then take the longest of these strings.
		goop = response.xpath('//script/text()').extract() + response.xpath('//style/text()').extract()

		# Find blurb using Welcome
		# html = response.xpath('//*[contains(text(), "Welcome")]')
		WELCOME_SELECTOR = '//*[contains(text(), "Welcome")]'
		# WELCOME_SELECTOR = '//*[contains(text(), "Welcome") or contains(text(), "welcome")]' # picks up too many news stories (eg we were glad to welcome)
		html = response.xpath(WELCOME_SELECTOR)
		welcome_blurbs = []
		for x in html:   # for each example of Welcome found in the html
			node = x.xpath('text()').extract()
			sibling = x.xpath('following-sibling::*/text()').extract()  ##descendant-or-self::*/
			full_list = node + sibling
			clean_string = make_clean_string(full_list, goop)
			welcome_blurbs.append(clean_string)
		if len(welcome_blurbs) > 0:
			fin_welcome_blurb = max(welcome_blurbs, key=len)
		else:
			fin_welcome_blurb = []
		
		# Find blurb using principal name

		principal_names = ['Headteacher', 'Headmaster', 'Headmistress', 'Head of School', 'Principal']
		failed_principal_blurbs = []
		found_blurb = False
		for principal_name in principal_names:  # for each possible principal name
			headteacher_1 = '//*[contains(.,"{principal}")]'
			html = response.xpath(headteacher_1.format(principal=principal_name)).extract()
			principal_blurbs = []
			for x in html:  # for each example of that principal name found in html
				headteacher_2 = '//*[contains(.,"{principal}")]/text() | ' \
						'//*[contains(.,"{principal}")]/preceding-sibling::*/text() | ' \
						'//*[contains(.,"{principal}")]/parent::*/preceding-sibling::*/text()'
				full_list = response.xpath(headteacher_2.format(principal=principal_name)).extract()
				clean_string = make_clean_string(full_list, goop)
				principal_blurbs.append(clean_string)
			if len(principal_blurbs)>0:
				fin_principal_blurb = max(principal_blurbs, key=len)  # longest blurb for a given principal name
			else:
				fin_principal_blurb = []

		fin_blurb = max([fin_welcome_blurb, fin_principal_blurb], key=len)
		if len(fin_blurb) == 0:
			flag = 'Found_url_but_no_blurb'
		else:
			flag = 'Found_blurb'
		result = dict(url=used_url, blurb=fin_blurb, length=len(fin_blurb), time=str(datetime.now()), 
			urn=urn, orig_url=orig_url, flag=flag)
		# result_df = DataFrame(result.items())
		# with open(FILENAME, 'a') as f:
		# 	result_df.to_csv(f, index=False, header=False, encoding='utf-8')
		yield result


