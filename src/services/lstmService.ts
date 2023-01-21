import * as tf from "@tensorflow/tfjs-node";
import { IPriceData } from "./currencyService";

export const createModel = async (dataLength: number) => {
    const model = tf.sequential();

    // kernelInitializer: Used to initialize the weight matrix that is used in the dot product between the input and the weights in the LSTM layer.
    // recurrentInitializer: Used to initialize the weight matrix that is used in the dot product between the recurrent state (i.e. the previous hidden state) and the weights in the LSTM layer.
    // returnSequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
    model.add(tf.layers.lstm({ units: 128, inputShape: [dataLength, 1], kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 64, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 32, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 64, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

    // SGD (Stochastic Gradient Descent) vs Adam (Adaptive Moment Estimation) vs RMSprop (Root Mean Square Propagation) optimizers:
    /* SGD updates the model parameters by moving in the opposite direction of the gradient of the loss function, with a fixed learning rate. It is relatively simple to implement and understand, but can be sensitive to the choice of learning rate. */
    /* Adam is a more advanced optimization algorithm that builds on SGD by incorporating information about the second moments of the gradients (i.e. the variance) in addition to the first moments (i.e. the mean). This allows Adam to automatically adjust the learning rate for each parameter, which can lead to faster convergence and better performance. However, it is also more computationally expensive than SGD. */
    /* RMSprop (Root Mean Square Propagation) is an optimizer that uses the root mean square of the gradient of the weights to scale the learning rate. It is similar to the Adagrad optimizer but uses a moving average of the squared gradient instead of the accumulation of all previous squared gradients. RMSprop is known to work well on recurrent neural networks and is often used in combination with other optimizers. */
    // model.compile({ optimizer: "adam", loss: 'meanSquaredError' });

    // console.log("Model compiled!");
    return model;
}

export const trainModel = async (model: tf.Sequential, trainingData: IPriceData[], labelData: IPriceData[]) => {
    // console.log("Stating model training...");
    // console.info("Training data length:", trainingData.length);
    // console.info("Label data length:", labelData.length);

    // Array of prices.
    const inputData = trainingData.map((priceData) => priceData.price);

    // Normalize the data.
    const mean = inputData.reduce((a, b) => a + b, 0) / inputData.length;
    const std = Math.sqrt(inputData.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / inputData.length);
    const normalizedInputData = inputData.map(x => (x - mean) / std);
    // console.log("normalized training data:", normalizedInputData);

    // Target output price.
    const labels = labelData.map((priceData) => priceData.price);
    // console.log("label:", labels);

    // Convert trainingData and labels to tensors
    const xs = tf.tensor(normalizedInputData, [1, normalizedInputData.length, 1]);
    // console.info("xs shape:", xs.shape);

    const ys = tf.tensor([labels], [1, labels.length, 1]);
    // console.info("ys shape:", ys.shape);

    // Fit model to the data
    const history = await model.fit(xs, ys, {
            batchSize: 66,
            shuffle: true,
            epochs: 1000,
            
        }) 
        .catch((err) => {
            console.error("Error fitting model:", err);
        });
    // TODO: Look into installing a node backend: https://github.com/tensorflow/tfjs-node for more details.

    console.info("history:", history);
    // console.log("Model finished training!");
}

export const modelPredict = async (model: tf.Sequential, inputData: IPriceData[]) => {
    // console.log("Stating model prediction.");
    // console.log("Input data length:", inputData.length);

    // Array of prices.
    const input = inputData.map((priceData) => priceData.price);
    // console.info("input:", input);

    // Convert input to tensor
    const xs = tf.tensor(input, [1, input.length, 1]);
    // console.info("xs shape:", xs.shape);

    // Make prediction
    const prediction = model.predict(xs) as tf.Tensor;
    console.info("prediction:", prediction);

    // Extract the predicted value from the tensor
    const predictedValue = prediction.dataSync()[0];

    return predictedValue;
}

/*

For your specific case, with data ranging from 2021-01-16 to 2023-01-15, you could use the following split:

Training set: Use the data from 2021-01-16 to 2022-09-15, about 700*0.8 = 560 data points for training.
Validation set: Use the data from 2022-09-16 to 2022-11-15, about 700*0.1 = 70 data points for validation.
Test set: Use the data from 2022-11-16 to 2023-01-15, about 700*0.1 = 70 data points for testing.

---

const model = tf.sequential();

const lstmLayer = tf.layers.lstm({
    units: 8, // Number of LSTM units in the layer
    inputShape: [1, 6], // The shape of the input data, in this case, [batchSize, sequenceLength, 6]
});

model.add(lstmLayer);

model.compile({
    optimizer: 'adam',
    loss: 'meanSquaredError'
});

This example creates a sequential model and adds a single LSTM layer with 8 units. The input shape of the data is [batchSize, sequenceLength, 6]. The model is then compiled with the 'adam' optimizer and the 'meanSquaredError' loss function.

Once the model is created, you can use the fit method to train the model on the training data. For example:

const xs = tf.tensor3d(trainingData); // trainingData should be a 3D tensor of shape [batchSize, sequenceLength, 6]
const ys = tf.tensor2d(trainingLabels); // trainingLabels should be a 2D tensor of shape [batchSize, 1]

// Train the model for a specific number of epochs
await model.fit(xs, ys, {
    epochs: 10,
    validationData: [tf.tensor3d(validationData), tf.tensor2d(validationLabels)],
});

In this example, trainingData should be a 3D tensor of shape [batchSize, sequenceLength, 6], with 560 data points and trainingLabels should be a 2D tensor of shape [batchSize, 1] with the corresponding labels.

The validationData should also be a 3D tensor of shape [batchSize, sequenceLength, 6], with 70 data points and validationLabels should be a 2D tensor of shape [batchSize, 1] with the corresponding labels.

Keep in mind that, this is a simple example, you can adjust the number of layers, the number of units, the batch size, the number of epochs and other parameters to optimize the model's performance.

---

Please analyse the code below and let me know if you see any issues with it. I am trying to predict the price of a cryptocurrency using a LSTM model.
I am using the Tensorflow.js library.
At the bottom of I've included the history log from the model training. I am not sure if the loss is too low / high or if the model is underfitting / overfitting.

```
const makeAI = async (traininData, labelData, actualData, dataLength) => {
    const model = tf.sequential();
    
    model.add(tf.layers.lstm({ units: 128, inputShape: [dataLength, 1], kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 64, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 32, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.01 }), recurrentRegularizer: tf.regularizers.l2({ l2: 0.01 })}));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 64, kernelInitializer: 'glorotUniform', recurrentInitializer: 'glorotUniform', returnSequences: true }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.dense({ units: 1 }));
    model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });
    
    // Array of prices.
    const inputData = trainingData.map((priceData) => priceData.price);
    
    // Normalize the data.
    const mean = inputData.reduce((a, b) => a + b, 0) / inputData.length;
    const std = Math.sqrt(inputData.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b, 0) / inputData.length);
    const normalizedInputData = inputData.map(x => (x - mean) / std);
    
    // Target output price.
    const labels = labelData.map((priceData) => priceData.price);
    
    // Convert trainingData and labels to tensors
    const xs = tf.tensor(normalizedInputData, [1, normalizedInputData.length, 1]);
    const ys = tf.tensor([labels], [1, labels.length, 1]);
    
    // Fit model to the data
    const history = await model.fit(xs, ys, {
            batchSize: 66,
            shuffle: true,
            epochs: 100,
        }) 
        .catch((err) => {
            console.error("Error fitting model:", err);
        });
    
    // Let the model make a prediction.
    const input = actualData.map((priceData) => priceData.price);
    
    // Convert input to tensor
    const xs = tf.tensor(input, [1, input.length, 1]);
    
    const prediction = model.predict(xs) as tf.Tensor;
    const predictedValue = prediction.dataSync()[0];
    
    return predictedValue;
}
```

Logs from model training:

History {
  validationData: null,
  params: {
    epochs: 100,
    initialEpoch: 0,
    samples: 1,
    steps: null,
    batchSize: 66,
    verbose: 1,
    doValidation: false,
    metrics: [ 'loss' ]
  },
  epoch: [
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99
  ],
  history: {
    loss: [
      1661186432, 1661185792, 1661185152, 1661184384, 1661183488,    
      1661182336, 1661180800, 1661179008, 1661176576, 1661172992,    
      1661168000, 1661160576, 1661149568, 1661135232, 1661112960,    
      1661083904, 1661044096, 1660995584, 1660947840, 1660903040,    
      1660850432, 1660823680, 1660791040, 1660766848, 1660735360,    
      1660710528, 1660684800, 1660660736, 1660633728, 1660609024,    
      1660584448, 1660561792, 1660548480, 1660528000, 1660510208,    
      1660496256, 1660485120, 1660472320, 1660459904, 1660449024,    
      1660441856, 1660433280, 1660421120, 1660414720, 1660406656,    
      1660398208, 1660393984, 1660385408, 1660381568, 1660372224,    
      1660363776, 1660351232, 1660349440, 1660340864, 1660335232,    
      1660322816, 1660325888, 1660312960, 1660310144, 1660305408,    
      1660299776, 1660292224, 1660283136, 1660278912, 1660269696,    
      1660269568, 1660260992, 1660250112, 1660251904, 1660243200,    
      1660236672, 1660222848, 1660224256, 1660222720, 1660217728,    
      1660203392, 1660202496, 1660192384, 1660193408, 1660186624,    
      1660173696, 1660172928, 1660161024, 1660157696, 1660162944,    
      1660146688, 1660138624, 1660130688, 1660134784, 1660120448,    
      1660122624, 1660104448, 1660111616, 1660102528, 1660092288,    
      1660093824, 1660085376, 1660074624, 1660074240, 1660064640     
    ]
  }
}
Approximate number we're trying to predict: 19941.780543296303
Model prediction: 1.2185672521591187
Model took: 657345.8761999905ms

*/