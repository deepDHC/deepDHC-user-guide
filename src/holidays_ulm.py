from datetime import timedelta
from workalendar.europe.germany import BadenWurttemberg

class UlmHolidays:

    def __init__(self):
        self.cal = BadenWurttemberg()

    def get_carneval_days(self, year):
        days = []
        easter_sunday = self.cal.get_easter_sunday(year)
        carnival_days = [46, 47, 48, 52]
        for day in carnival_days:
            day = (easter_sunday - timedelta(days=day))
            days.append(day)
        return days
    
    def get_all_holidays(self, year):
        days = self.cal.get_calendar_holidays(year)
        fd = self.get_carneval_days(year)
        for d in fd:
            days.append((d, 'Carnival'))
        return days
