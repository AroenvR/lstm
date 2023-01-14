import * as tf from "@tensorflow/tfjs";
import { getBtcData, IOHLCData } from "./services/currencyService";
import { writeFile } from "./services/fileService";
import { compileModel, modelPredict, trainModel } from "./services/lstmService";

// Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
const ai = async () => {

    // This example creates a stacked LSTM model with 2 layers, the first layer has 10 units and input shape of [30, 4] and returnSequences is set to true because it's the first layer and in the output of this layer we would like to use it as input for the next layer. The second layer has 8 units. The LSTM is then followed by a dense layer with 4 units, one for each of the open, high, low, and close values. The model is then compiled with the Adam optimizer and mean squared error as the loss function. The training data is preprocessed so that the last 31 days of data is used as the target values, and all other data is used as input values. Finally, the model is trained on the synthetic data using 10 epochs.

    // Define the model architecture
    const model = await compileModel();
    await trainModel(model);

    console.log(await modelPredict(model, 420)); // 840

    // console.log("Model:", model);

    // Prepare the training data
    let numOfDays = 7; // Maximum is 334
    const trainingData: void | IOHLCData[] = await getBtcData(numOfDays + 31)
        .then ((data) => {
            writeFile(`./trainingDataFor${numOfDays}.json`, JSON.stringify(data));
            return data;
        })
        .catch((err) => {
            console.log("Failed getting training data. Status:", err.response.status, "Message:", err.message);
        }) as IOHLCData[];

    // The input data will be an array of OHLC values with a length equal to numOfDays.
    // The expected output data will be a single OHLC value 31 days into the future.

    // So I need to extract the numOfDays from the data and make it an array.
    // Then I need to get 31 indexes later, the value of the the expected output.





    
    // const tensor = convertToTensor(trainingData);
    // const { inputs, labels } = tensor;
    
    // await trainModel(model, inputs, labels);

    // console.log("trainingData:", trainingData);


    // Every thing works until here, but then ChatGPT's code doesn't work below.
    // Need to refactor the existing code to become a better stack, and to accept my input data.
    // Then I need to train the model and figure out how to check it's output (if it works).



    // // Create a new model with lstm Layer
    // const LSTM = tf.layers.lstm({units: 4, returnSequences: true});
    
    // // Create a 3d tensor
    // const x = tf.tensor([1, 2, 3, 4], [2, 2, 1]);
    
    // // Apply lstm layer to x
    // const output = LSTM.apply(x);
    // console.log("Model output:", output.print());





    // const output = model.apply(x);
 
    // // Print output
    // output.print()

    // const xs = tf.tensor3d(trainingData.slice(0, -31).map(({open, high, low, close}) => [open, high, low, close]));
    // const ys = tf.tensor3d(trainingData.slice(-31).map(({open, high, low, close}) => [open, high, low, close]));
    // console.log(`got an xs: ${xs} and a ys: ${ys}`);

    // // Train the model
    // await model.fit(xs, ys, { epochs: 10 });


        /*
    const flatXs = trainingData.slice(0, -31).map(({open, high, low, close}) => [open, high, low, close]).flat();
    console.log("Got a flatXs");

    const flatYs = trainingData.slice(-31).map(({open, high, low, close}) => [open, high, low, close]).flat();
    console.log("Got a flatYs");

    const xs = tf.tensor3d(flatXs, [flatXs.length / 4, 30, 4]);
    const ys = tf.tensor3d(flatYs, [flatYs.length / 4, 31, 4]);

    console.log(`got an xs: ${xs} and a ys: ${ys}`);

    // Train the model
    await model.fit(xs, ys, { epochs: 10 });
    */

    console.log("Done training!");
}
ai();

/*
    This example creates an LSTM with 10 units and input shape of [10, 1]. The LSTM is then followed by a dense layer with 1 unit. The model is then compiled with the Adam optimizer and mean squared error as the loss function. Finally, the model is trained on synthetic data using 10 epochs.

    const tf = require('@tensorflow/tfjs');

    // Define the model architecture
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 10, inputShape: [10, 1] }));
    model.add(tf.layers.dense({ units: 1 }));

    // Compile the model
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    // Generate some synthetic data for training
    const xs = tf.tensor2d(
    Array.from({ length: 100 }, (_, i) => [...Array(10)].map((_, j) => i + j)
    );
    const ys = tf.tensor2d(
    Array.from({ length: 100 }, (_, i) => [i + 10])
    );

    // Train the model
    await model.fit(xs, ys, { epochs: 10 });

*/












// runFooApi();








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