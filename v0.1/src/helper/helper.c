#include "helper.h"

#include <assert.h>
#include <time.h>

#include "livero_types.h"

uint64_t lv_convert_date_to_epoch(const LVDate* date) {
    LVDate d = {0};
    if (date) {
        assert(lv_date_is_valid(date));
        d = *date;
    } else {
        time_t current_time;
        time(&current_time);

        struct tm utc;
        gmtime_r(&current_time, &utc);

        d.year = utc.tm_year + 1900;
        d.month = utc.tm_mon + 1;
        d.day = utc.tm_mday;
        d.hour = utc.tm_hour;
        d.min = utc.tm_min;
        d.sec = utc.tm_sec;
    }

    uint32_t year = date->year;
    uint32_t month = date->month;
    uint32_t day = date->day;

    // Treat the year as starting on March 1st, so Feb 29 falls at the very end.
    if (month <= 2) {
        year -= 1;
    }

    // Month index counted from March: Mar=0, ..., Jan=10, Feb=11.
    uint32_t mp = (month > 2) ? month - 3 : month + 9;

    // 0-based day of the March-based year: [0, 365].
    uint32_t doy = (153 * mp + 2) / 5 + (day - 1);

    // Days from 0000-03-01 to this date.
    uint64_t days = 365ULL * year + year / 4 - year / 100 + year / 400 + doy;

    // 719468 = days from 0000-03-01 to 1970-01-01.
    days -= 719468;

    return days * 86400ULL + (uint64_t)date->hour * 3600 + (uint64_t)date->min * 60 + date->sec;
}

void lv_convert_epoch_to_date(const uint64_t epoch_time, LVDate* date) {
    date->sec = epoch_time % 60;
    uint64_t remaining_minutes = epoch_time / 60;

    date->min = remaining_minutes % 60;
    uint64_t remaining_hours = remaining_minutes / 60;

    date->hour = remaining_hours % 24;
    uint64_t days = remaining_hours / 24;

    days += 719468;

    // 3. 400years = 146097days
    uint64_t era = days / 146097;
    uint32_t doe = days % 146097;  // era [0, 146096]

    uint32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;  // era [0, 399]
    uint32_t y = yoe + era * 400;
    uint32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);  // 0, 365]

    uint32_t mp = (5 * doy + 2) / 153;           // [0, 11]
    uint8_t day = doy - (153 * mp + 2) / 5 + 1;  // [1, 31]
    uint8_t month = mp < 10 ? mp + 3 : mp - 9;   // [1, 12]

    date->year = y + (month <= 2);
    date->month = month;
    date->day = day;
}

int lv_date_is_valid(const LVDate* date) {
    assert(date);

    if (date->month < 1 || date->month > 12) {
        return 0;
    }

    if (date->day < 1 || date->day > 31) {
        return 0;
    }

    const uint8_t time_values[] = {date->hour, date->min, date->sec};

    for (LVSize32_t i = 0; i < sizeof(time_values) / sizeof(time_values[0]); ++i) {
        const uint8_t t_value = time_values[i];
        if (t_value > 59) {
            return 0;
        }
    }

    return 1;
}
