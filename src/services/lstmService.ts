import * as tf from "@tensorflow/tfjs";
import { IPriceData } from "./currencyService";

export const createModel = async (dataLength: number) => {
    const model = tf.sequential();

    // kernelInitializer: Used to initialize the weight matrix that is used in the dot product between the input and the weights in the LSTM layer.
    // recurrentInitializer: Used to initialize the weight matrix that is used in the dot product between the recurrent state (i.e. the previous hidden state) and the weights in the LSTM layer.
    // returnSequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    model.add(tf.layers.lstm({ units: 64, inputShape: [dataLength, 1], kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.lstm({ units: 32, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.lstm({ units: 16, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.lstm({ units: 32, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.lstm({ units: 64, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.lstm({ units: 32, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.lstm({ units: 16, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.lstm({ units: 8, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.dense({ units: 1 }));

    // SGD (Stochastic Gradient Descent) vs Adam (Adaptive Moment Estimation) vs RMSprop (Root Mean Square Propagation) optimizers:
    /* SGD updates the model parameters by moving in the opposite direction of the gradient of the loss function, with a fixed learning rate. It is relatively simple to implement and understand, but can be sensitive to the choice of learning rate. */
    /* Adam is a more advanced optimization algorithm that builds on SGD by incorporating information about the second moments of the gradients (i.e. the variance) in addition to the first moments (i.e. the mean). This allows Adam to automatically adjust the learning rate for each parameter, which can lead to faster convergence and better performance. However, it is also more computationally expensive than SGD. */
    /* RMSprop (Root Mean Square Propagation) is an optimizer that uses the root mean square of the gradient of the weights to scale the learning rate. It is similar to the Adagrad optimizer but uses a moving average of the squared gradient instead of the accumulation of all previous squared gradients. RMSprop is known to work well on recurrent neural networks and is often used in combination with other optimizers. */
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

    console.log("Model compiled!");
    return model;
}

export const trainModel = async (model: tf.Sequential, trainingData: IPriceData[], labelData: IPriceData[]) => {
    console.log("Stating model training...");
    console.info("Training data length:", trainingData.length);
    console.info("Label data length:", labelData.length);

    // Array of prices.
    const inputData = trainingData.map((priceData) => priceData.price);
    // console.log("trainingData:", trainingData);

    // Target output price.
    const labels = labelData.map((priceData) => priceData.price);
    // console.log("label:", labels);

    // Convert trainingData and labels to tensors
    const xs = tf.tensor(inputData, [1, inputData.length, 1]);
    // console.info("xs shape:", xs.shape);

    const ys = tf.tensor([labels], [1, labels.length, 1]);
    // console.info("ys shape:", ys.shape);

    // Fit model to the data
    await model.fit(xs, ys, { epochs: 25 }) // TODO: Look into installing a node backend: https://github.com/tensorflow/tfjs-node for more details.
        .catch((err) => {
            console.error("Error fitting model:", err);
        });

    console.log("Model finished training!");
}

export const modelPredict = async (model: tf.Sequential, inputData: IPriceData[]) => {
    console.log("Stating model prediction.");
    console.log("Input data length:", inputData.length);

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