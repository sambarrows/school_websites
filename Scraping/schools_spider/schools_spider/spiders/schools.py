# -*- coding: utf-8 -*-
from scrapy import Spider
from scrapy.http import Request
import re
from urlparse import urljoin as origurljoin

def check_if_url(url_try):
	if not isinstance(url_try, basestring):
		success = False
	else:
		if url_try == 'http://':
			success = False
		else:
			success = True
	return success

class SchoolsSpider(Spider):
	name = "schools"
	allowed_domains = ["www.education.gov.uk"]
	max_pages = 1204
	start_urls = (
			'http://www.education.gov.uk/edubase/public/quickSearchResult.xhtml?page=1',
	)

	def start_requests(self):
		for i in range(1,self.max_pages+1):
			# print 'http://www.education.gov.uk/edubase/public/quickSearchResult.xhtml?page='+str(i)
			yield Request('http://www.education.gov.uk/edubase/public/quickSearchResult.xhtml?page='+str(i), callback=self.parse)

	def parse(self, response):
		summary_urls = response.xpath('//*[@class="search_results"]//a[@title="View establishment"]/@href').extract()
		for summary_url in summary_urls:
			yield Request(response.urljoin(summary_url), self.parse_summarypage)

	def parse_summarypage(self, response):
		urn = response.xpath('//h1[@class="edUrnLeft"]/text()').extract_first()
		school_url = response.xpath('//th[contains(., "Website")]/following-sibling::td/div/a/@href').extract_first()
		if check_if_url(school_url)==False:
			base_url = re.findall('^.*(?=summary)', response.url)[0]  ##must create new base url, as urljoin won't work with summary_url (which is now the response.url)
			end_url = response.xpath('//*[contains(text(), "Details")]/../@href').extract_first()
			# end_url = response.xpath('//*[@class="menu"]//li[2]/a/@href').extract_first()  ##how to select by "Details" in span, rather than position?
			details_url = origurljoin(base_url, end_url)
			return Request(details_url, self.parse_detailspage)
		else:
			return{
				'urn': urn,
				'school_url': school_url
			}

	def parse_detailspage(self, response):
		urn = response.xpath('//h1[@class="edUrnLeft"]/text()').extract_first()  # what if wasn't on details page? How to yield form two levels?
		school_url = response.xpath('//*[contains(text(), "Website")]/following-sibling::td//a/@href').extract_first()
		if check_if_url(school_url) == False:
			school_url = ''
		return {
			'urn': urn,
			'school_url': school_url
		}