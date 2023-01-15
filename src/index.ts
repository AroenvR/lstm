import * as tf from "@tensorflow/tfjs";
import { getCryptoData, IOHLCData, IPriceData } from "./services/currencyService";
import { writeFile } from "./services/fileService";
import { createModel, modelPredict, trainModel } from "./services/lstmService";
import { prepareCryptoTrainingData } from "./services/trainingService";
import { isTruthy } from "./util/util";
import btc_input_data from "./json_files/btc_input_data.json";
import btc_label_data from "./json_files/btc_label_data.json";
import btc_prediction_data from "./json_files/btc_prediction_data.json";

// Hi, looks like you are running TensorFlow.js in Node.js. To speed things up dramatically, install our node backend, visit https://github.com/tensorflow/tfjs-node for more details. 
const ai = async () => {
    console.log("Stating.");
    const start = performance.now();

    // const { trainingData, targetData } = await prepareCryptoTrainingData("bitcoin", 720, 1611532800000, 1654041600000);

    // Define the model architecture
    const model = await createModel(btc_input_data.length);

    await trainModel(model, btc_input_data, btc_label_data);

    const prediction = await modelPredict(model, btc_prediction_data);
    console.log("Number we're trying to predict:", 19941.780543296303);
    console.log("Model prediction:", prediction);

    console.log("Model took:", performance.now() - start, "ms");
    
    /* The current code works, but it's got a shit prediction. Prediction should be as close to label as possible. 
    Logs:
        Model compiled!
        Stating model training...
        trainingData: [
            number,
            number,
            number,
        ]
        label: 19407.447430262473
        Model finished training!
        Training took: 2.605909001082182 ms
        Stating model prediction.
        prediction: 0
    */

    /*
        The prediction output being 0 suggests that the model is not able to make accurate predictions. There could be several reasons for this:

        The training data may not be representative of the problem you are trying to solve. The input data may not contain enough information for the model to learn from, or it may not be the right type of data for the problem.

        The model architecture may not be suitable for the problem. The LSTM layer with 1 unit and input shape of [dataLength, 1] may not have enough capacity to learn the underlying patterns in the data.

        The model may not have been trained for enough time. With only 1000000 epochs, the model may not have had enough time to converge to a good solution.

        The optimization algorithm and loss function used may not be appropriate for the problem. 'sgd' and 'meanSquaredError' may not be the best choices for this specific problem

        The input data preprocessing and normalization may not have been done properly.

        It's important to note that without more information about the problem and the data, it's difficult to say for certain why the model is not making accurate predictions. It would be necessary to investigate further by visualizing the data, experimenting with different model architectures, and tweaking the training parameters to improve the model's performance.
    */
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