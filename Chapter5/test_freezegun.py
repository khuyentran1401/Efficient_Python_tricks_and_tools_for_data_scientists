from freezegun import freeze_time
import datetime 

def get_day_of_week():
    return datetime.datetime.now().weekday()

@freeze_time("2024-10-13")
def test_get_day_of_week():
    assert get_day_of_week() == 6
