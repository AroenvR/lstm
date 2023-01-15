# Cryptocurrency data files for training an LSTM

## 720_daily_bitcoin.json 
Contains the full data set for 720 days on a daily chart.

## btc_input_data.json
Contains the input data for training the AI

## btc_label_data.json
Contains the label data for training the AI -> label data is expected output

## btc_prediction_data.json
Contains the data we'll give the AI to make a prediction.

---

The data was scalped in the following way:  
For the input data, the last 2 months were removed from the data set.  
For the ouput data, the first month and the last month were removed from the data set.  

This results in the input data starting at 2021-01-25T00:00:00.000Z  
The label data starts at 2021-02-24T00:00:00.000Z  
The prediction data starts at 2021-02-24T00:00:00.000Z

The input data ends at 2022-11-15T00:00:00.000Z  
And the label data ends at 2022-12-15T00:00:00.000Z 
The prediction data ends at 2022-12-15T00:00:00.000Z   
Real data set ends at 2023-01-14T00:00:00.000Z

This allows us to have the label data be 1 month further ahead from the input data, and grants us the ability to hide 1 month from the AI's future.

// TODO: Get data for 730 days (meth is hard).