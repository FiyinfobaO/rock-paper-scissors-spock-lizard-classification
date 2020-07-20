let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
const dataset = new RPSDataset();
var rockSamples=0, paperSamples=0, scissorsSamples=0, spockSamples=0, lizardSamples=0;
let isPredicting = false;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');  // loading mobile net
  const layer = mobilenet.getLayer('conv_pw_13_relu');  // to select a layer from mobilenet
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
  dataset.ys = null;
  dataset.encodeLabels(5);  
  // create the new model
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 125, activation: 'relu'}),
      tf.layers.dense({ units: 5, activation: 'softmax'})
    ]
  });
  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['acc']});
  model.fit(dataset.xs, dataset.ys, {
    epochs: 30,
    callbacks: {
      onEpochEnd: async(epoch, logs) =>{
                                      console.log("Epoch: " + epoch + " Loss: " + logs.loss + " Accuracy: " + logs.acc);
                                  }
      }
   });
}

function handleButton(elem){
	switch(elem.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;
        case "3":
			spockSamples++;
			document.getElementById("spocksamples").innerText = "Spock samples:" + spockSamples;
			break;            
        case "4":
			lizardSamples++;
			document.getElementById("lizardsamples").innerText = "Lizard samples:" + lizardSamples;
			break;
	}
	label = parseInt(elem.id);  //extracts the label by converting the id to int
	const img = webcam.capture();  // captures webcam contents
	dataset.addExample(mobilenet.predict(img), label);  // adds the prediction of the image from mobile net alongside the labels to the dataset

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();  // captures an image frame from the webcam
      const activation = mobilenet.predict(img);  // Get the immediate activation from MobileNet
      const predictions = model.predict(activation);  // Pass the activations to the retrained model to get predictions as probabilities.
      return predictions.as1D().argMax();  // one hot encoding after which the highest prediction index value is gotten
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
		case 2:
			predictionText = "I see Scissors";
			break;
        case 3:
			predictionText = "I see Spock";
			break;
        case 4:
			predictionText = "I see Lizard";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;  // displays the prediction text
			
    
    predictedClass.dispose();  //to dispose predicted class to free memory
    await tf.nextFrame();  // to prevent locking up the ui thread so the page stays responsive
  }
}


function doTraining(){
	train();
    alert("Training Done!")
}

function predictMe(){
	isPredicting = true;  // allows us to do continous predictions
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	mobilenet = await loadMobilenet();  // gets mobilenet model
	tf.tidy(() => mobilenet.predict(webcam.capture()));  // warms up the model to prevents any lags
		
}



init();
