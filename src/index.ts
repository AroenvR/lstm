import * as tf from "@tensorflow/tfjs";
import { getCryptoData, IOHLCData, IPriceData } from "./services/currencyService";
import { writeToFile } from "./services/fileService";
import { createModel, modelPredict, trainModel } from "./services/lstmService";
import { prepareCryptoTrainingData } from "./services/trainingService";
import { isTruthy } from "./util/util";
import btc_input_data from "./json_files/btc_input_data.json";
import btc_label_data from "./json_files/btc_label_data.json";
import btc_prediction_data from "./json_files/btc_prediction_data.json";
import eth_prediction_data from "./json_files/eth_prediction_data.json";

// Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
const ai = async () => {
    console.log("Stating.");
    const start = performance.now();

    // const { trainingData, targetData } = await prepareCryptoTrainingData("bitcoin", 720, 1611532800000, 1654041600000);


    // Create and compile a model:
    const model = await createModel(btc_input_data.length);

    // Train the model:
    await trainModel(model, btc_input_data, btc_label_data);

    // Make a prediction:
    const prediction = await modelPredict(model, btc_prediction_data);
    console.log("Number we're trying to predict (BTC):", 19941.780543296303);
    // console.log("Number we're trying to predict (ETH):", 1549.111474081304);
    console.log("Model prediction:", prediction);

    console.log(`Model took: ${performance.now() - start}ms`);

    // Saving the model:
    // const saveResults = await model.save('file://path/to/my/model', {include_optimizer: true, save_format: 'tfjs'});

    // Loading a model:
    // const loadedModel = await tf.loadLayersModel('file://path/to/my/model');
}
ai();

const getData = async () => {
    const data = await getCryptoData("ethereum", 730);
    console.log(data);

    writeToFile("730_daily_eth.json", data);
}
// getData();






// Different project:
/*
    1. Get Input:   - number of shares
                    - number of threshold
                    - keys (list)

    2. Generate randomized hash. 
     2.1: Generate a completely random number. <- will need to be improved later, so make it scalable.
     2.2: Hash that completely random number using crypto_hash_sha512 from https://nacl.cr.yp.to/hash.html

    3. Split the hash into n shares using Shamir's Secret Sharing (https://www.geeksforgeeks.org/shamirs-secret-sharing-algorithm-cryptography/) may help.

    4. Encrypt the shares with the keys provided using crypto_stream_aes128ctr from https://nacl.cr.yp.to/stream.html .

    5. Return the CipherShares.

*/