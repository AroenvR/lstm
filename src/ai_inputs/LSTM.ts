import axios from 'axios';

/**
 * Simple GET Request.
 * @param url addition to the base domain url to send the getRequest to.
 * @returns server's response object if response.ok, else returns void.
 * Response.ok: https://developer.mozilla.org/en-US/docs/Web/API/Response/ok
 */
const httpGet = async (url: string): Promise<void | object | any>=> {
    return await axios.get(`https://${url}`)
        .then((response) => {
            if (response.status === 200) {
                return response.data;
            }
        })
        .catch((error) => {
            console.error("Error in httpGet: ", error);
            console.error("Error code: ", error.code);
            console.error("httpGet: error.response.status", error.response.status);
            console.error("httpGet: error.response.statusText", error.response.statusText);
            throw error;
        });
}

/**
 * Interface for OHLC data.
 * @interface IOHLCData
 * @property {number} time - Unix timestamp in milliseconds.
 * @property {string} date - Date in ISO format.
 * @property {number} open - Opening price.
 * @property {number} high - Highest price.
 * @property {number} low - Lowest price.
 * @property {number} close - Closing price.
 */
interface IOHLCData {
    time: number;
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
}

/**
 * Get Bitcoin data from coingecko.com
 * @param numOfDays number of days to get data for. Max is 720.
 * @returns Promise<IOHLCData[]> - Array of IOHLCData.
 * @throws Error if response is undefined.
 *  @example
 * [
 *  {
 *      time: 1673712000000,
 *      date: 2023-01-14T16:00:00.000Z,
 *      open: 20738.59,
 *      high: 20738.59,
 *      low: 20738.59,
 *      close: 20738.59
 *  },
 * ]
 */
const getBtcData = async (numOfDays: number): Promise<IOHLCData[]> => {
    const url = `api.coingecko.com/api/v3/coins/bitcoin/ohlc?vs_currency=usd&days=${numOfDays}`;
    const response = await httpGet(url).catch((error) => {
        console.error("getBtcData: error: ", error);
        throw error;
    });

    const formattedResponse: IOHLCData[] = response.map((item: number[]) => {
        /* [{ time: 1673712000000, date: 2023-01-14T16:00:00.000Z, open: 20738.59, high: 20738.59, low: 20738.59, close: 20738.59 }] */
        return {
            time: item[0],
            date: new Date(item[0]),
            open: item[1],
            high: item[2],
            low: item[3],
            close: item[4],
        };
    });
    
    return formattedResponse;
}

const ai = async () => {
    // This example creates a stacked LSTM model with 2 layers, the first layer has 10 units and input shape of [30, 4] and returnSequences is set to true because it's the first layer and in the output of this layer we would like to use it as input for the next layer. The second layer has 8 units. The LSTM is then followed by a dense layer with 4 units, one for each of the open, high, low, and close values. The model is then compiled with the Adam optimizer and mean squared error as the loss function. The training data is preprocessed so that the last 31 days of data is used as the target values, and all other data is used as input values. Finally, the model is trained on the synthetic data using 10 epochs.

    const tf = require('@tensorflow/tfjs');

    // Define the model architecture
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 10, inputShape: [30, 4], returnSequences: true }));
    model.add(tf.layers.lstm({ units: 8 }));
    model.add(tf.layers.dense({ units: 4 }));

    // Compile the model
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    // Prepare the training data
    const trainingData: void | IOHLCData[] = await getBtcData(7)
        .catch((err) => {
            console.log("Failed getting training data. Status:", err.response.status, "Message:", err.message);
        }) as IOHLCData[];

    console.info("Successfully got training data. Length:", trainingData?.length);

    console.log("type of training data:", typeof trainingData) // will give you the shape of the data

    const xs = tf.tensor3d(trainingData.slice(0, -31).map(({open, high, low, close}) => [open, high, low, close]));
    const ys = tf.tensor3d(trainingData.slice(-31).map(({open, high, low, close}) => [open, high, low, close]));
    console.log(`got an xs: ${xs} and a ys: ${ys}`);

    // Train the model
    await model.fit(xs, ys, { epochs: 10 });
    console.log("Done training!");
}
ai();