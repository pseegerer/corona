import os
import urllib.request
from datetime import datetime, timedelta

import click
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import slack
from pandas import to_datetime as dt


def date2int(X):
    return np.array([datetime.toordinal(x) for x in X]).reshape((-1, 1))


def coef2factor(coef):
    return np.exp(coef)


def download_and_read_data(country):
    # Download data
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
    df_file = "data.csv"
    urllib.request.urlretrieve(url, filename=df_file)

    # Read data
    df = pd.read_csv(df_file)
    not_time = ['Province/State', 'Country/Region', 'Lat', 'Long']
    df_country = df[df["Country/Region"] == country].drop(not_time, axis=1).T

    df_country.index = dt(df_country.index)

    return df_country


def compute_confidence_intervals(y_pred, X_pred, df_ger, confidence=95):
    residuals = np.log(
        pd.DataFrame(y_pred[:, 0], index=X_pred)[0] / df_ger.iloc[:,
                                                      0]).dropna()

    # CI under normality assumption
    num_stds = {
        68: 1,
        95: 2,
        99: 3,
    }[confidence]
    upper = np.exp(np.log(y_pred.flatten()) + num_stds * residuals.std())
    lower = np.exp(np.log(y_pred.flatten()) - num_stds * residuals.std())

    return lower, upper


@click.command()
@click.option("--country", default="Germany")
@click.option("--start_date", default="2020-03-14",
              help="Date when to start the regression.")
@click.option("--look_ahead", default=10, type=int,
              help="How many days to look into the future from today on.")
@click.option("--post", is_flag=True, help="Whether to post it to Slack.")
def main(country, start_date, look_ahead, post):
    # Parameters
    confidence = 95
    look_ahead = look_ahead * timedelta(days=1)

    df = download_and_read_data(country)
    message = ""
    message += f"Latest measurement from `{df.index[-1]}`\n"

    valid_indices = df.index >= dt(start_date)
    X = df[valid_indices].index
    X_int = date2int(X)
    y_log = np.log(df[valid_indices])

    # Fit model
    linreg = sklearn.linear_model.LinearRegression()
    linreg.fit(X_int, y_log)
    r2 = linreg.score(X_int, y_log)
    message += f"Fitted linear regression with R^2={r2:.3f}\n"

    factor = coef2factor(linreg.coef_[0, 0])
    message += f"Increase factor per day: {factor:.3f}\n"

    # In how many days does the number double?
    message += f"Number doubles every {np.log(2) / np.log(factor):.1f} days, multiplies by 10 every {np.log(10) / np.log(factor):.1f}.\n"

    # Predict
    X_pred = pd.date_range(start_date, end=datetime.today() + look_ahead,
                           freq='d')
    y_pred = np.exp(linreg.predict(date2int(X_pred)))

    # Compute confidence intervals
    lower, upper = compute_confidence_intervals(y_pred, X_pred, df,
                                                confidence=confidence)

    # Print
    fstr = "%a %d, %b"
    message += "```\n"
    message += "           Predict.      Diff.   Measur.    Diff.\n"
    message += "------------------------------------------------\n"
    y_prev = np.nan
    for day_curr, y, l, u in zip(X_pred, y_pred[:, 0], lower, upper):
        if day_curr.weekday() == 0:
            message += "\n"
        try:
            measurement = df.loc[day_curr.date()].values[0]
        except KeyError:
            measurement = np.nan

        # Difference to day before
        try:
            diff_measurement = measurement - df.loc[
                day_curr.date() - timedelta(days=1)].values[0]
        except KeyError:
            diff_measurement = np.nan

        diff_y = y - y_prev
        y_prev = y

        message += f"{day_curr.to_pydatetime().strftime(fstr)} {y:7.0f} ({diff_y:7.0f}+) {measurement:7.0f} ({diff_measurement:7.0f}+)"

        message += "\n"
    message += "```"

    # Plot
    plt.figure(figsize=(16, 16))
    for i, log in enumerate([True, False]):
        plt.subplot(2, 1, i + 1)
        plt.plot(df, label="Measurements", marker="|")
        plt.plot(X_pred, y_pred, label="Prediction", linestyle="--")
        plt.fill_between(X_pred, upper, y2=lower, alpha=0.2,
                         label=f"{confidence}% confidence interval")
        plt.axvline(x=dt(datetime.today().date()), color="grey", linestyle="--",
                    label="Today")
        plt.xlim([dt(start_date), X_pred[-1]])
        plt.ylim([y_pred[0], y_pred[-1]])
        plt.gca().xaxis.set_minor_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
        if log:
            plt.yscale("log")
        plt.legend(loc=2)
        plt.suptitle(f"Increase factor per day: {factor:.3f}")

    if post:
        # Post to Slack
        plt.savefig("corona_regression.png")
        with open(os.environ["SLACK_TOKEN"], "r") as f:
            slack_token = f.read().strip()
        client = slack.WebClient(token=slack_token)
        response = client.chat_postMessage(
            channel='#corona',
            text=message)
        response = client.files_upload(
            channels='#corona',
            file="corona_regression.png")
    else:
        print(message)
        plt.show()


if __name__ == '__main__':
    main()