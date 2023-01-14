import { isTruthy } from "../util/util";
import { getCryptoData, IOHLCData, IPriceData } from "./currencyService";

/**
 * Training data interface.
 * @interface ICryptoTrainingData
 * @property IPriceData[] dataChunk - training data.
 * @property IPriceData targetData - the data to predict.
 */
export interface ICryptoTrainingData {
    trainingData: IPriceData[];
    targetData: IPriceData;
}

/**
 * Get the training data.
 * @param currency - currency to get data for.
 * @param numOfDays - number of days to get data for. Max is 360.
 * @param fromEpoch - epoch to start chunking from.
 * @param toEpoch - epoch to end chunking at.
 * @returns Promise<ICryptoTrainingData> - data chunk (training data) and target data (the data to predict).
 * @throws Error if response is undefined or toEpoch is too large.
 */
export const prepareCryptoTrainingData = async (currency: string, numOfDays: number, fromEpoch: number, toEpoch: number): Promise<ICryptoTrainingData> => {
    // Get the training data
    const trainingData: void | IOHLCData[] | IPriceData[] = await getCryptoData(currency, numOfDays)
        .then ((data) => {
            // writeFile(`./trainingDataFor${numOfDays}${currency}.json`, data);
            return data;
        })
        .catch((err) => {
            console.log("Failed getting training data. Status:", err.response.status, "Message:", err.message);
        }) as IPriceData[];
    console.info("Last allowed epoch:", trainingData[trainingData.length - 32].time);

    if (toEpoch > trainingData[trainingData.length - 32].time) {
        console.error("toEpoch is too large. Max is 31 days from last allowed epoch.");
        console.error("toEpoch:", toEpoch);
        console.error("Last allowed epoch:", trainingData[trainingData.length - 32].time);
        throw new Error("trainingData (service): toEpoch is too large. Max is 31 days from last allowed epoch.");
    }

    // Format and chunk the data
    return chunkData(trainingData, fromEpoch, toEpoch);
}

/**
 * Chunk the data into a data chunk and a target data.
 * @param data - data to chunk.
 * @param fromEpoch - epoch to start chunking from.
 * @param toEpoch - epoch to end chunking at.
 * @returns ICryptoTrainingData - data chunk (training data) and target data (the data to predict).
 * @throws Error if dataChunk or targetData is undefined.
 */
const chunkData = (data: IPriceData[], fromEpoch: number, toEpoch: number): ICryptoTrainingData => {
    let dataChunks: IPriceData[] = [];
    let targetData: IPriceData | undefined;

    for (let i = 0; i < data.length; i++) {
        if (data[i].time >= fromEpoch && data[i].time <= toEpoch) {
            dataChunks.push(data[i]);
        }
        if (data[i].time >= toEpoch + 2678400000) { // 2678400000 = 31 days in milliseconds
            targetData = data[i];
            break;
        }
    }

    if (!isTruthy(dataChunks) || !isTruthy(targetData)) {
        console.error("dataChunks:", dataChunks);
        console.error("targetData:", targetData);
        throw new Error("chunkData: Failed chunking data.");
    }

    return { trainingData: dataChunks, targetData: targetData! };
}