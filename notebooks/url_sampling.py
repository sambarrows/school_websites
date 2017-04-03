import webbrowser
import pandas as pd
import random

# Read in cleaned and prepared urls
url_dat = pd.read_csv('../Scraping/csvurls.csv', names=['URN','URL'])

# Randomly select 100 urls (s)
index_vals = range(len(url_dat))
random.seed(a=1.0) # et seed so can replicated analysis
random.shuffle(index_vals)
url_dat = url_dat.ix[index_vals[:4]]
url_dat.reset_index(drop=True, inplace=True)

# Open a url in turn, categorize it, then move onto the next url
# (can't get Python to close a Chrome tab, so I have to do this manually as I go)
blurb_type_list = []
for i in range(range(len(url_dat))):
    url = url_dat.loc[i,'URL']
    print url
    chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
    webbrowser.get(chrome_path).open(url)
    blurb_type = raw_input('Blurb type (1=welcome, 2=head, 3=welcome+head, 4=other_blurb, 5=no_blurb, 6=broken_url: ')
    blurb_type_list.append(blurb_type)
    
# Attach blurb categorizations to data frame
url_dat['blurb_type'] = pd.Series( blurb_type_list, index=url_dat.index)

# Write to csv
url_dat.to_csv('../blurb_cats.csv')