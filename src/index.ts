import { runFooApi } from "./api/fooApi";
import { getBtcData, IOHLCData } from "./services/currencyService";
import { httpGet } from "./services/httpService";
import { isTruthy } from "./util/util";

// Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
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
    console.log("Model compiled!");
    // console.info("Model: ", model);
    console.info("Is model training: ", model.isTraining);

    // Prepare the training data
    let numOfDays = 365;
    const trainingData: void | IOHLCData[] = await getBtcData(numOfDays)
        .catch((err) => {
            console.log("Failed getting training data. Status:", err.response.status, "Message:", err.message);
        }) as IOHLCData[];

    console.log(`Successfully got training data for ${numOfDays} days!`);
    console.info("trainingData: ", trainingData);

    // const x = tf.tensor([1, 2, 3, 4], [2, 2,1]);
    // console.log("x: ", x);

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