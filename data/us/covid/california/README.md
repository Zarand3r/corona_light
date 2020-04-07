# california-coronavirus-data

The Los Angeles Times' independent tally of coronavirus cases in California.

## About the data

These files come from a continual Times survey of California's 58 county health agencies and three city agencies. Updated numbers are published throughout the day at [latimes.com/coronavirustracker](https://www.latimes.com/projects/california-coronavirus-cases-tracking-outbreak/). This repository will periodically update with an extract of that data.

The figures are typically ahead of the totals compiled by California's Department of Public Health. By polling local agencies, The Times database also gathers some information not provided by the state. The system has [won praise](https://twitter.com/palewire/status/1243329930877788160) from public health officials, who do not dispute its method of data collection.

The tallies here are mostly limited to residents of California, the standard method used to count patients by the state’s health authorities. Those totals do not include people from other states who are quarantined in California, such as the passengers and crew of the Grand Princess cruise ship that docked in Oakland.

## Reusing the data

The Los Angeles Times is making coronavirus infections data available for use by researchers and scientists to aid in the fight against COVID-19.

The company's Terms of Service apply. By using the data, you accept and agree to follow the [Terms of Services](https://www.latimes.com/terms-of-service).

It states that "you may use the content online only, and solely for your personal, non-commercial use, provided you do not remove any trademark, copyright or other notice from such Content," and that, "no other use is permitted without prior written permission of Los Angeles Times."

Reselling the data is forbidden. Any use of these data in published works requires attribution to the Los Angeles Times.

To inquire about reuse, please contact Data and Graphics Editor Ben Welsh at [ben.welsh@latimes.com](mailto:ben.welsh@latimes.com).

## Data dictionary

### [latimes-agency-totals.csv](./latimes-agency-totals.csv)

The total cases and deaths logged by local public health agencies each day. Each row contains the cumulative totals reported by a single agency as of that date.

Most counties have only one agency except for Alameda and Los Angeles counties, where some cities run independent health departments. In Alameda County, the city of Berkeley is managed independently. In Los Angeles County, Pasadena and Long Beach are managed independently. These cities' totals are broken out into separate rows. In order to calculate county-level totals, you must aggregate them together using the `county` field.

| field           | type    | description                                                                                                                                                                                                                           |
|:----------------|:--------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `agency`          | string  | The name of the county or city public health agency that provided the data. Guaranteed to be unique when combined with the `date` field.                                                                                             |
| `county`          | string  | The name of the county where the agency is based.                                                                                                                                                                                     |
| `fips`            | string  | The [FIPS code](https://en.wikipedia.org/wiki/Federal_Information_Processing_Standards) given to the `county` by the federal government. Can be used to merge with other data sources.                                                  |
| `date`            | date    | The date when the data were retrieved in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.                                                                                                                                                                                                |
| `confirmed_cases` | integer | The cumulative number of confirmed coronavirus cases as of this `date`.                                                                                                                                                                     |
| `deaths`          | integer | The cumulative number of deaths attributed to coronavirus as of this `date`.                                                                                                                                                               |
| `did_not_update`  | boolean | Indicates if the agency did not provide an update on this date. If this is `true` and the case and death totals are unchanged from the previous day, this means they were holdovers. Use this flag omit these records when desired. |

### [latimes-place-totals.csv](./latimes-place-totals.csv)

Some counties, primarily in Southern California, break out the location of cases within their service area. The Times is gathering and consolidating these lists. Each row contains cumulative case totals reported in that area as of that date.

These counties currently provide the data and are being tracked by The Times:

* Los Angeles
* Kern
* Orange
* Riverside
* San Bernardino
* San Diego
* San Luis Obispo
* Santa Barbara
* Ventura

The locations provided by Los Angeles County correspond to the public health department's official "Countywide Statistical Areas". Locations in other counties are manually geocoded by The Times.

Be aware that some counties have shifted the place names used over time.

In some circumstances the true total of cases is obscured. Los Angeles and Orange counties decline to provide the precise number of cases in areas with low populations and instead provide a potential range. The lowest number in the range is entered into the record in the `confirmed_cases` field and an accompanying `note` includes the set of possible values.

| field           | type    | description                                                                                                                                                                          |
|-----------------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `date`            | date    | The date when the data were retrieved in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.                                                                                                                                               |
| `county`          | string  | The name of the county where the city is located.                                                                                                                                    |
| `fips`            | string  | The [FIPS code](https://en.wikipedia.org/wiki/Federal_Information_Processing_Standards) given to the county by the federal government. Can be used to merge with other data sources. |
| `place`           | string  | The name of the city, neighborhood or other area.                                                                                                                                    |
| `confirmed_cases` | integer | The cumulative number of confirmed coronavirus case at that time.                                                                                                                    |
| `note`      | string  | In cases where the `confirmed_cases` are obscured, this explains the range of possible values.                                                                                         |
| `x`               | float   | The longitude of the `place`.                                                                                                                                        |
| `y`               | float   | The latitude of the `place`.                                                                                                                                         |

### [cdph-state-totals.csv](./cdph-state-totals.csv)

Totals published by the California Department of Public Health in [its daily press releases](https://www.cdph.ca.gov/Programs/OPA/Pages/New-Release-2020.aspx). Each row contains all numbers included in that day's release.

This file includes cases and deaths, the age of victims, transmission types, tests and hospitalizations. The state has stopped supplying some fields and began supplying some others over time. Deprecated fields have been maintained in this file and are noted below.

Due to bureaucratic lags, the state's totals for cases and deaths arrive slower than The Times numbers, which are generated by surveying local agencies. The state's methods for collecting testing data have struggled with [errors](https://www.latimes.com/california/story/2020-04-04/gavin-newsom-california-coronavirus-testing-task-force) and [delays](https://www.latimes.com/business/story/2020-03-30/its-taking-up-to-eight-days-to-get-coronavirus-tests-results-heres-why).

Some dates are missing because the state did not publish a press release for that day.

| field                      | type    | description                                                                                                                                                                                                  |
|:---------------------------|:--------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `date`                       | date    | The date when the data were retrieved in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.                                                                                                                                                                        |
| `confirmed_cases`            | integer | The cumulative number of confirmed coronavirus case at that time.                                                                                                                                            |
| `deaths`                     | integer | The cumulative number of deaths at that time.                                                                                                                                                                |
| `travel`                     | integer | The number of cases acquired while traveling. The state stopped publishing it on March 24.                                                                                                                   |
| `person_to_person`           | integer | The number of cases acquired from close contacts and family members. The state stopped publishing it on March 24.                                                                                            |
| `community_spread`           | integer | The number of cases acquired from community spread. The state stopped publishing it on March 29.                                                                                                             |
| `under_investigation`        | integer | The number of cases acquired from an unknown source. The state stopped publishing it on March 24.                                                                                                            |
| `other_causes`               | integer | The number of cases acquired from other sources. On March 24 the state began combining this figure with travel, person-to-person and under investigation cases. On March 29 the state stopped publishing it. |
| `self_monitoring`            | integer | The number of people in a form of isolation monitored by the state. The state stopped publishing it on March 19.                                                                                             |
| `age_0_to_17`                | integer | The number of confirmed cases involving a person between 0 and 18 years old.                                                                                                                                 |
| `age_18_to_49`               | integer | The number of confirmed cases involving a person between 18 and 49 years old.                                                                                                                                |
| `age_50_to_64`               | integer | The number of confirmed cases involving a person between 50 and 64 years old.                                                                                                                                |
| `age_65_and_up`              | integer | The number of confirmed cases involving a person 65 of older.                                                                                                                                                |
| `age_18_to_64`               | integer | The number of confirmed cases involving a person between 18 and 64 years old. The state stopped publishing it on March 23.                                                                                   |
| `age_unknown`                | integer | The number of confirmed cases involving a person of unknown age.                                                                                                                                             |
| `total_tests`                | integer | The total number of tests conducted.                                                                                                                                                                         |
| `received_tests`             | integer | The number of tests results received.                                                                                                                                                                        |
| `pending_tests`              | integer | The number of tests resuts that are still pending.                                                                                                                                                           |
| `confirmed_hospitalizations` | integer | The total number of hospitalizations with a confirmed case of COVID-19.                                                                                                                                      |
| `confirmed_icu`              | integer | The number of ICU hospitalizations with a confirmed case of COVID-19.                                                                                                                                        |
| `suspected_hospitalizations` | integer | The total number of hospitalizations with a suspected case of COVID-19.                                                                                                                                      |
| `suspected_icu`              | integer | The number of ICU hospitalizations with a suspected case of COVID-19.                                                                                                                                        |
| `source_url`                 | string  | The URL to the state press release.                                                                                                                                                                    |

## Contact

To inquire about the data or about reuse, please contact Data and Graphics Editor [Ben Welsh](https://palewi.re/who-is-ben-welsh/) at [ben.welsh@latimes.com](mailto:ben.welsh@latimes.com)
