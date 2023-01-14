import * as tf from "@tensorflow/tfjs";
import { timeStamp } from "console";
import { IOHLCData } from "./currencyService";

export const compileModel = async () => {
    const model = tf.sequential();

    // From: https://www.youtube.com/watch?v=Y_XM3Bu-4yc
    // Linear Regression model that takes an input of 1 and an output of 1
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    // Prepare for training
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' }); // Stochastic Gradient Descent

    // From: https://codelabs.developers.google.com/codelabs/tfjs-training-regression#6 
    // // Add a single input layer
    // model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));

    // // Add an output layer
    // model.add(tf.layers.dense({units: 1, useBias: true}));

    // the code above works, the code below was GPT's code but it doesn't work

    // Chat GPT layers:
    // model.add(tf.layers.lstm({ units: 10, inputShape: [30, 4], returnSequences: true }));
    // model.add(tf.layers.lstm({ units: 8 }));
    // model.add(tf.layers.dense({ units: 4 }));

    // Compile the model ?
    // model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

    console.log("Stacked LSTM Model compiled!");

    return model;
}

export const trainModel = async (model: tf.Sequential) => {
    // { number: 1 }, { number: 2 }, { number: 3 }, { number: 4 }, { number: 5 }
    const xs = tf.tensor1d([1, 2, 3, 4, 5]); // x = actual training features for the dataset.
    const ys = tf.tensor1d([2, 4, 6, 8, 10]); // y = label for the data. The model should predict this value.

    // Train the model
    await model.fit(xs, ys, { epochs: 1000 });

    console.log("Stacked LSTM Model trained!");
}

export const modelPredict = async (model: tf.Sequential, value: any) => {
    const output = model.predict(tf.tensor2d([value], [1, 1])) as tf.Tensor;
    const prediction = Array.from(output.dataSync())[0];

    return prediction;
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