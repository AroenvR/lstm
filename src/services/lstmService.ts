import * as tf from "@tensorflow/tfjs";
import { isTruthy } from "../util/util";
import { IPriceData } from "./currencyService";

export const createModel = async (dataLength: number) => {
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 1, inputShape: [dataLength, 1] })); // 7 time steps, 1 feature
    
    // SGD (Stochastic Gradient Descent) vs Adam (Adaptive Moment Estimation) optimizers:
    /* SGD updates the model parameters by moving in the opposite direction of the gradient of the loss function, with a fixed learning rate. It is relatively simple to implement and understand, but can be sensitive to the choice of learning rate. */
    /* Adam is a more advanced optimization algorithm that builds on SGD by incorporating information about the second moments of the gradients (i.e. the variance) in addition to the first moments (i.e. the mean). This allows Adam to automatically adjust the learning rate for each parameter, which can lead to faster convergence and better performance. However, it is also more computationally expensive than SGD. */
    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    console.log("Model compiled!");
    return model;
}

export const trainModel = async (model: tf.Sequential,inputData: IPriceData[], targetOutput: IPriceData) => {
    console.log("Stating model training...");
    console.info("data length:", inputData.length);

    // Array of prices.
    const trainingData = inputData.map((priceData) => priceData.price);
    console.log("trainingData:", trainingData);

    // Target output price.
    const label = targetOutput.price;
    console.log("label:", label);

    // Convert trainingData and label to tensors
    const xs = tf.tensor(trainingData, [1, trainingData.length, 1]);
    // console.info("xs shape:", xs.shape);

    const ys = tf.tensor([label], [1, 1]); // batch size, time steps, features
    // console.info("ys shape:", ys.shape);

    // Fit model to the data
    model.fit(xs, ys, { epochs: 1000000000 })
        .catch((err) => {
            console.error("Error fitting model:", err);
        });

    console.log("Model finished training!");
}

export const modelPredict = async (model: tf.Sequential, inputData: IPriceData[]) => {
    console.log("Stating model prediction.");

    // Array of prices.
    const input = inputData.map((priceData) => priceData.price);
    // console.info("input:", input);

    // Convert input to tensor
    const xs = tf.tensor(input, [1, input.length, 1]);
    // console.info("xs shape:", xs.shape);

    // Make prediction
    const prediction = model.predict(xs) as tf.Tensor;
    // console.info("prediction:", prediction);

    // Extract the predicted value from the tensor
    const predictedValue = prediction.dataSync()[0];

    return predictedValue;
}



// /**
//  * 
//  * @param data 
//  */
// export const convertToTensor = (data: IOHLCData[]) => {
//     return tf.tidy(() => {
//         // Shuffle the data
//         tf.util.shuffle(data);

//         // Convert data to Tensor
//         const inputs = data.map((obj) => obj.open);
//         const labels = data.map((obj) => obj.close);

//         const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
//         const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

//         // Normalize the data to the range 0 - 1 using min-max scaling
//         const inputMax = inputTensor.max();
//         const inputMin = inputTensor.min();
//         const labelMax = labelTensor.max();
//         const labelMin = labelTensor.min();

//         const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
//         const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

//         return {
//             inputs: normalizedInputs,
//             labels: normalizedLabels,
//             // Return the min/max bounds so we can use them later.
//             inputMax,
//             inputMin,
//             labelMax,
//             labelMin,
//         }
//     });
// }

// /**
//  * 
//  * @param model 
//  * @param inputs 
//  * @param labels 
//  * @returns 
//  */
// export const trainModel = async (model: tf.Sequential, inputs: tf.Tensor, labels: tf.Tensor) => {
//     // Prepare the model for training.
//     model.compile({
//         optimizer: tf.train.adam(),
//         loss: tf.losses.meanSquaredError,
//         metrics: ['mse'],
//     });

//     const batchSize = 43;
//     const epochs = 50;

//     return await model.fit(inputs, labels, {
//         batchSize,
//         epochs,
//         shuffle: true,
//       });
// }

/**
 * **DOES NOT WORK**
 * @param model 
 * @param inputData 
 * @param normalizationData 
 */
// export const testModel = (model: tf.Sequential, inputData: IOHLCData[], normalizationData: any) => {
//     const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

//     // Generate predictions for a uniform range of numbers between 0 and 1;
//     // We un-normalize the data by doing the inverse of the min-max scaling that we did earlier.
//     const [xs, preds] = tf.tidy(() => {

//         const xsNorm = tf.linspace(0, 1, 100);
//         const predictions = model.predict(xsNorm.reshape([100, 1]));
    
//         const unNormXs = xsNorm
//           .mul(inputMax.sub(inputMin))
//           .add(inputMin);
    
//         const unNormPreds = predictions
//           .mul(labelMax.sub(labelMin))
//           .add(labelMin);
    
//         // Un-normalize the data
//         return [unNormXs.dataSync(), unNormPreds.dataSync()];
//     });
    
//     const predictedPoints = Array.from(xs).map((val, i) => {
//         return {x: val, y: preds[i]}
//     });
    
//     const originalPoints = inputData.map(d => ({
//         x: d.open, y: d.close,
//     }));

//     console.log("originalPoints", originalPoints);
//     console.log("predictedPoints", predictedPoints);
// }


// Given this code:

// // Add a single input layer
// model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

// // Add an output layer
// model.add(tf.layers.dense({units: 1, useBias: true}));

// // the code above works, the code below was GPT's code but it doesn't work
// // model.add(tf.layers.lstm({ units: 10, inputShape: [30, 4], returnSequences: true }));
// // model.add(tf.layers.lstm({ units: 8 }));
// // model.add(tf.layers.dense({ units: 4 }));




/*
Prompt:


The following code:

```
export const createModel = async (dataLength: number) => {
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 1, inputShape: [dataLength, 1] })); // 7 time steps, 1 feature
    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    console.log("Model compiled!");
    return model;
}

export const trainModel = async (model: tf.Sequential,inputData: IPriceData[], targetOutput: IPriceData) => {
    console.log("Stating model training...");
    const startOfTraining = performance.now();

    // console.info("data length:", inputData.length);

    // Array of prices.
    const trainingData = inputData.map((priceData) => priceData.price);
    console.log("trainingData:", trainingData);

    // Target output price.
    const label = targetOutput.price;
    console.log("label:", label);

    // Convert trainingData and label to tensors
    const xs = tf.tensor(trainingData, [1, trainingData.length, 1]);
    // console.info("xs shape:", xs.shape);

    const ys = tf.tensor([label], [1, 1]); // batch size, time steps, features
    // console.info("ys shape:", ys.shape);

    // Fit model to the data
    model.fit(xs, ys, { epochs: 1000000 })
        .catch((err) => {
            console.error("Error fitting model:", err);
        });

    console.log("Model finished training!");
    console.log("Training took:", performance.now() - startOfTraining, "ms");
}

export const modelPredict = async (model: tf.Sequential, inputData: IPriceData[]) => {
    console.log("Stating model prediction.");

    // Array of prices.
    const input = inputData.map((priceData) => priceData.price);
    // console.info("input:", input);

    // Convert input to tensor
    const xs = tf.tensor(input, [1, input.length, 1]);
    // console.info("xs shape:", xs.shape);

    // Make prediction
    const prediction = model.predict(xs) as tf.Tensor;
    // console.info("prediction:", prediction);

    // Extract the predicted value from the tensor
    const predictedValue = prediction.dataSync()[0];

    return predictedValue;
}
```

Provides the following log:
Model compiled!
Stating model training...
trainingData: [
  47335.42029920565,
  48823.05520964915,
  49338.78465183332,
  49934.5282762881,
  50013.02994694588,
  51696.206431317885,
  52739.800166345514
]
label: 53894.59599452107
Model finished training!
Training took: 2.605909001082182 ms
Stating model prediction.
prediction: 0

















Help me make a model prediction function using JavaScript and TensorFlow.

export const createModel = async (dataLength: number) => {
    const model = tf.sequential();
    model.add(tf.layers.lstm({ units: 1, inputShape: [dataLength, 1] })); // 7 time steps, 1 feature
    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

    console.log("Model compiled!");
    return model;
}

export const trainModel = async (inputData: IPriceData[], targetOutput: IPriceData) => {
    console.log("data length:", inputData.length);

    const model = await createModel(inputData.length);

    // Array of prices.
    const trainingData = inputData.map((priceData) => priceData.price);
    console.log("trainingData:", trainingData);

    // Target output price.
    const label = targetOutput.price;
    console.log("label:", label);

    // Convert trainingData and label to tensors
    const xs = tf.tensor(trainingData, [1, trainingData.length, 1]);
    console.log("xs shape:", xs.shape);

    const ys = tf.tensor([label], [1, 1]); // batch size, time steps, features
    console.log("ys shape:", ys.shape);

    // Fit model to the data
    model.fit(xs, ys, { epochs: 1000 })
        .catch((err) => {
            console.log("Error fitting model:", err);
        });

    console.log("Model trained!");
}

The log output is:
Model compiled!
trainingData: [
  47335.42029920565,
  48823.05520964915,
  49338.78465183332,
  49934.5282762881,
  50013.02994694588,
  51696.206431317885,
  52739.800166345514
]
label: 53894.59599452107
xs shape: [ 1, 7, 1 ]
ys shape: [ 1, 1 ]
Model trained!




















The following code:

```
console.log("data length:", inputData.length);

const model = tf.sequential(); // file line 48
model.add(tf.layers.lstm({ units: 8, inputShape: [7, 1] }));

model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });

// Array of prices.
const trainingData = inputData.map((priceData) => priceData.price);
console.log("trainingData:", trainingData);

// Target output price.
const label = targetOutput.price;
console.log("label:", label);


// Convert trainingData and label to tensors
const xs = tf.tensor2d(trainingData, [trainingData.length, 1]).reshape([1, trainingData.length, 1]);
console.log("xs shape:", xs.shape);

const ys = tf.tensor2d([label], [1, 1]);
console.log("ys shape:", ys.shape);

// Fit model to the data
model.fit(xs, ys, { epochs: 100 })
    .catch((err) => {
        console.log("Error fitting model:", err);
    }); // file line 73
```

Has the following console output:

data length: 7
trainingData: [
    47335.42029920565,
    48823.05520964915,
    49338.78465183332,
    49934.5282762881,
    50013.02994694588,
    51696.206431317885,
    52739.800166345514
]
label: 53894.59599452107
xs shape: [ 1, 7, 1 ]
ys shape: [ 1, 1 ]
Error fitting model: ValueError: target expected a batch of elements where each example has shape [8] (i.e.,tensor shape [*,8]) but the target received an input with 1 examples, each with shape [1] (tensor shape [1,1])
    at new ValueError (/home/aroenvr/projects/crypto_ai/tfjs-layers/src/errors.ts:48:5)
    at standardizeInputData (/home/aroenvr/projects/crypto_ai/tfjs-layers/src/engine/training.ts:170:17)
    at LayersModel.standardizeUserDataXY (/home/aroenvr/projects/crypto_ai/tfjs-layers/src/engine/training.ts:1169:9)
    at LayersModel.<anonymous> (/home/aroenvr/projects/crypto_ai/tfjs-layers/src/engine/training.ts:1194:14)
    at step (/home/aroenvr/projects/crypto_ai/node_modules/tslib/tslib.es6.js:102:23)
    at Object.next (/home/aroenvr/projects/crypto_ai/node_modules/tslib/tslib.es6.js:83:53)
    at /home/aroenvr/projects/crypto_ai/node_modules/tslib/tslib.es6.js:76:71
    at new Promise (<anonymous>)
    at __awaiter (/home/aroenvr/projects/crypto_ai/node_modules/tslib/tslib.es6.js:72:12)
    at LayersModel.standardizeUserData (/home/aroenvr/projects/crypto_ai/node_modules/@tensorflow/tfjs-layers/dist/tf-layers.node.js:23899:16)

Please help me fix the error.






*/