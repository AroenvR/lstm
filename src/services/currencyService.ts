import { httpGet } from './httpService';

/**
 * Interface for OHLC data.
 * @interface IOHLCData
 * @property number time - Unix timestamp in milliseconds.
 * @property string date - Date in ISO format.
 * @property number open - Opening price.
 * @property number high - Highest price.
 * @property number low - Lowest price.
 * @property number close - Closing price.
 */
export interface IOHLCData {
    time: number;
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
}

/**
 * Interface for price data.
 * @interface IPriceData
 * @property {number} time - Unix timestamp in milliseconds.
 * @property {string} date - Date in ISO format.
 * @property {number} price - Price.
 * @property {number} marketCap - Market cap.
 * @property {number} totalVolume - Total volume.
 */
export interface IPriceData {
    time: number;
    date: string;
    price: number;
    marketCap: number;
    totalVolume: number;
}

/**
 * Get Bitcoin data from coingecko.com
 * @param numOfDays number of days to get data for. Max is 360.
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
export const getCryptoData = async (currency: string, numOfDays: number): Promise<IOHLCData[] | IPriceData[]> => {
    const url = `api.coingecko.com/api/v3/coins/${currency}/market_chart?vs_currency=usd&days=${numOfDays}&interval=daily`;
    const response = await httpGet(url).catch((error) => {
        console.error("getBtcData: error: ", error);
        throw error;
    });
    
    const prices = response.prices.map((item: number[]) => item[1]);
    const marketCaps = response.market_caps.map((item: number[]) => item[1]);
    const totalVolumes = response.total_volumes.map((item: number[]) => item[1]);

    let cryptoData: IPriceData[] = [];
    for (let i = 0; i < prices.length - 1; i++) {
        cryptoData.push({
            time: response.prices[i][0],
            date: new Date(response.prices[i][0]).toISOString(),
            price: prices[i],
            marketCap: marketCaps[i],
            totalVolume: totalVolumes[i],
        });
    }

    return cryptoData;

    // const formattedResponse: IOHLCData[] = response.map((item: number[]) => {
    //     /* [{ time: 1673712000000, date: 2023-01-14T16:00:00.000Z, open: 20738.59, high: 20738.59, low: 20738.59, close: 20738.59 }] */
    //     return {
    //         time: item[0],
    //         date: new Date(item[0]),
    //         open: item[1],
    //         high: item[2],
    //         low: item[3],
    //         close: item[4],
    //     };
    // });
    // console.log("formattedResponse:", formattedResponse);

    // return formattedResponse;
}