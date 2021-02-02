import scrapy

class FightSpider(scrapy.Spider):
    name = "fight_scraper"

    def start_requests(self):
        """Start at all events page."""
        url = "http://ufcstats.com/statistics/events/completed?page=all"
        yield scrapy.Request(url=url, callback=self.parse_lvl1)

    def parse_lvl1(self, response):
        """Visit eachh event link from all events page."""
        links = response.css('i.b-statistics__table-content a::attr(href)').getall()
        for l in links[1:]:
            yield scrapy.Request(url=l, callback=self.parse_lvl2)
    
    def parse_lvl2(self, response):
        """Visit each fight from the event."""
        date = response.css('li.b-list__box-list-item::text').getall()[1].strip()
        links = response.css('tr::attr(data-link)').getall()
        for l in links:
            yield scrapy.Request(url=l, callback=self.parse_lvl3, cb_kwargs=dict(date=date))
    
    def parse_lvl3(self, response, date):
        """Extract fight data."""
        totals_stats = ['total_str', 'td', 'td_pct', 'sub', 'rev', 'ctrl']
        sig_str_stats = ['sig_str', 'sig_str_pct', 'head', 'body', 'leg', 'distance', 'clinch', 'ground']       
        fighters = ['f1', 'f2']
        
        extract = lambda x: response.xpath(x).get().strip()

        stats = {}
        stats['date'] = date
        stats['event_title'] = extract('/html/body/section/div/h2/a/text()')
        stats['f1'] = extract('/html/body/section/div/div/div[1]/div[1]/div/h3/a/text()')
        stats['f2'] = extract('/html/body/section/div/div/div[1]/div[2]/div/h3/a/text()')
        stats['win_method'] = extract('/html/body/section/div/div/div[2]/div[2]/p[1]/i[1]/i[2]/text()')
        stats['round'] = response.css('i.b-fight-details__text-item::text')[1].get().strip()
        stats['time'] = response.css('i.b-fight-details__text-item::text')[3].get().strip()
        stats['weight_class'] = response.css('.b-fight-details__fight-title ::text').getall()[-1].strip()
        
        f1_win_status = extract('/html/body/section/div/div/div[1]/div[1]/i/text()') 
        if f1_win_status == 'D':
            stats['winner'] = 'draw'
        elif f1_win_status == 'W':
            stats['winner'] = 'f1'
        else:
            stats['winner'] = 'f2'

        for f_idx, fighter in enumerate(fighters):
            for s_idx, stat in enumerate(totals_stats):
                stat_name = fighter + '_' + stat
                f_xp_num, s_xp_num = f_idx+1, s_idx+5
                stat_val = response.xpath(f'/html/body/section/div/div/section[2]/table/tbody/tr/td[{s_xp_num}]/p[{f_xp_num}]/text()').get().strip()
                stats[stat_name] = stat_val 
        
            for s_idx, stat in enumerate(sig_str_stats):
                    stat_name = fighter + '_' + stat
                    f_xp_num, s_xp_num = f_idx+1, s_idx+2
                    stat_val = response.xpath(f'/html/body/section/div/div/table/tbody/tr/td[{s_xp_num}]/p[{f_xp_num}]/text()').get().strip()
                    stats[stat_name] = stat_val 

        yield stats
            

