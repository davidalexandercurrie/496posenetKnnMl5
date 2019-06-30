// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/* ===
ml5 Example
KNN Classification on Webcam Images with poseNet. Built with p5.js
=== */
let video;
var counter = 0;
var confidences = [];
// Create a KNN classifier
const knnClassifier = ml5.KNNClassifier();
let poseNet;
let poses = [];
var oldPoses = [];
var classWithHighestScore = "A";
var oldResult;
var startTraining = false;
var trainingTimer = 0;
var trainingText = "";

// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// SAMPLES
let sample1;
let sample2;
let sample3;
let sample4;
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
function preload() {
  sample1 = loadSound("samples/sample1.wav");
  sample2 = loadSound("samples/sample2.wav");
  sample3 = loadSound("samples/sample3.wav");
  sample4 = loadSound("samples/sample4.wav");
  sample5 = loadSound("samples/sample5.wav");
}
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------------------

function setup() {
  const canvas = createCanvas(640, 480);
  canvas.parent("videoContainer");
  video = createCapture(VIDEO);
  video.size(width, height);
  // Create the UI buttons
  createButtons();
  // Create a new poseNet method with a single detection
  poseNet = ml5.poseNet(video, modelReady);
  // This sets up an event that fills the global variable "poses"
  // with an array every time new poses are detected
  poseNet.on("pose", function(results) {
    poses = results;
  });
  // Hide the video element, and just show the canvas
  video.hide();
}

function draw() {
  if (startTraining) {
    select("#status").html(trainingText);
  }
  image(video, 0, 0, width, height);
  // We can call both functions to draw all keypoints and the skeletons
  drawKeypoints();
  drawSkeleton();
  audioEngine();
  autoTrain();
  counter++;
}

function autoTrain() {
  if (startTraining === false) {
    counter = 0;
  }
  if (startTraining === true && counter % 60 === 0) {
    trainingTimer++;
    if (trainingTimer > 5) {
      if (trainingTimer <= 10) {
        addExample("A");
        trainingText = "Training Class A";
      } else if (trainingTimer <= 15) {
        addExample("B");
        trainingText = "Training Class B";
      } else if (trainingTimer <= 20) {
        addExample("C");
        trainingText = "Training Class C";
      } else if (trainingTimer <= 25) {
        addExample("D");
        trainingText = "Training Class D";
      } else if (trainingTimer <= 30) {
        addExample("E");
        trainingText = "Training Class E";
      } else {
        startTraining = false;
        trainingTimer = 0;
        select("#status").html("Training Complete... Click Play When Ready");
      }
    }
  }
  if (startTraining === true) {
    if (trainingTimer <= 5) {
      fill(0);
      textSize(64);
      select("#status").html("Training in... " + (5 - trainingTimer));
    }
  }
}

function audioEngine() {
  if (typeof confidences["A"] !== "undefined") {
    sample1.setVolume(confidences["A"] / 2, 0.3);
  }

  if (typeof confidences["B"] !== "undefined") {
    sample2.setVolume(confidences["B"] / 2, 0.3);
  }

  if (typeof confidences["C"] !== "undefined") {
    sample3.setVolume(confidences["C"] / 2, 0.3);
  }

  if (typeof confidences["D"] !== "undefined") {
    sample4.setVolume(confidences["D"] / 2, 0.3);
  }

  if (typeof confidences["E"] !== "undefined") {
    sample5.setVolume(confidences["E"] / 2, 0.3);
  }
}

function modelReady() {
  // TODO
  select("#status").html("Click Train Model When Ready");
}

// Add the current frame from the video to the classifier
function addExample(label) {
  // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
  if (poses.length === 0) {
    poses = oldPoses;
  } else {
    oldPoses = poses;
  }
  const poseArray = poses[0].pose.keypoints.map(p => [
    p.score,
    p.position.x,
    p.position.y
  ]);
  console.log("adding example");
  // Add an example with a label to the classifier
  knnClassifier.addExample(poseArray, label);
  updateCounts();
}

// Predict the current frame.
function classify() {
  // Get the total number of labels from knnClassifier
  const numLabels = knnClassifier.getNumLabels();
  if (numLabels <= 0) {
    console.error("There are no examples in any label");
    return;
  }

  if (poses.length === 0) {
    poses = oldPoses;
  } else {
    oldPoses = poses;
  }

  // Convert poses results to a 2d array [[score0, x0, y0],...,[score16, x16, y16]]
  const poseArray = poses[0].pose.keypoints.map(p => [
    p.score,
    p.position.x,
    p.position.y
  ]);

  // Use knnClassifier to classify which label do these features belong to
  // You can pass in a callback function `gotResults` to knnClassifier.classify function
  knnClassifier.classify(poseArray, gotResults);
  // }
}

// util function to create UI buttons
function createButtons() {
  // // // When the A button is pressed, add the current frame
  // // // from the video with a label of "A" to the classifier
  // buttonA = select("#addClassA");
  // buttonA.mousePressed(function() {
  //   addExample("A");
  // });

  // // When the B button is pressed, add the current frame
  // // from the video with a label of "B" to the classifier
  // buttonB = select("#addClassB");
  // buttonB.mousePressed(function() {
  //   addExample("B");
  // });

  // buttonC = select("#addClassC");
  // buttonC.mousePressed(function() {
  //   addExample("C");
  // });
  // buttonD = select("#addClassD");
  // buttonD.mousePressed(function() {
  //   addExample("D");
  // });

  // // Reset buttons
  // resetBtnA = select("#resetA");
  // resetBtnA.mousePressed(function() {
  //   clearLabel("A");
  // });

  // resetBtnB = select("#resetB");
  // resetBtnB.mousePressed(function() {
  //   clearLabel("B");
  // });

  // resetBtnC = select("#resetC");
  // resetBtnC.mousePressed(function() {
  //   clearLabel("C");
  // });
  // resetBtnD = select("#resetD");
  // resetBtnD.mousePressed(function() {
  //   clearLabel("D");
  // });

  // Predict button
  // buttonPredict = select("#buttonPredict");
  // buttonPredict.mousePressed(classify);

  // Clear all classes button
  buttonClearAll = select("#clearAll");
  buttonClearAll.mousePressed(clearAllLabels);

  playButton = select("#playButton");
  playButton.mousePressed(() => {
    startAudio();
    classify();
  });

  playButton = select("#stopButton");
  playButton.mousePressed(() => {
    stopAudio();
  });

  // saveButton = select("#saveButton");
  // saveButton.mousePressed(() => {
  //   knnClassifier.save();
  // });

  // loadButton = select("#loadButton");
  // loadButton.mousePressed(() => {
  //   knnClassifier.load("myKnn.json", customModelReady);
  //   console.log("hello");
  // });

  trainButton = select("#trainButton");
  trainButton.mousePressed(trainMyModel);
}

function trainMyModel() {
  startTraining = true;
}

function customModelReady() {
  console.log("model loaded!");
}

// Show the results
function gotResults(err, result) {
  // Display any error
  if (err) {
    console.error(err);
  }

  // if (typeof result.label !== "undefined") {
  //   classWithHighestScore = result.label;
  // }
  if (typeof result === "undefined") {
    result = oldResult;
  } else {
    oldResult = result;
  }

  if (result.confidencesByLabel) {
    confidences = result.confidencesByLabel;

    // result.label is the label that has the highest confidence
    if (result.label) {
      select("#result").html(result.label);
      select("#confidence").html(`${confidences[result.label] * 100} %`);
    }

    select("#confidenceA").html(
      `${confidences["A"] ? confidences["A"] * 100 : 0} %`
    );
    select("#confidenceB").html(
      `${confidences["B"] ? confidences["B"] * 100 : 0} %`
    );
    select("#confidenceC").html(
      `${confidences["C"] ? confidences["C"] * 100 : 0} %`
    );
    select("#confidenceD").html(
      `${confidences["D"] ? confidences["D"] * 100 : 0} %`
    );
    select("#confidenceE").html(
      `${confidences["E"] ? confidences["E"] * 100 : 0} %`
    );
  }
  classify();
}

// Update the example count for each label
function updateCounts() {
  const counts = knnClassifier.getCountByLabel();

  select("#exampleA").html(counts["A"] || 0);
  select("#exampleB").html(counts["B"] || 0);
  select("#exampleC").html(counts["C"] || 0);
  select("#exampleD").html(counts["D"] || 0);
  select("#exampleE").html(counts["E"] || 0);
}

// Clear the examples in one label
function clearLabel(classLabel) {
  knnClassifier.clearLabel(classLabel);
  updateCounts();
}

// Clear all the examples in all labels
function clearAllLabels() {
  knnClassifier.clearAllLabels();
  updateCounts();
}

// A function to draw ellipses over the detected keypoints
function drawKeypoints() {
  // Loop through all the poses detected
  for (let i = 0; i < poses.length; i++) {
    // For each pose detected, loop through all the keypoints
    let pose = poses[i].pose;
    for (let j = 0; j < pose.keypoints.length; j++) {
      // A keypoint is an object describing a body part (like rightArm or leftShoulder)
      let keypoint = pose.keypoints[j];
      // Only draw an ellipse is the pose probability is bigger than 0.2
      if (keypoint.score > 0.2) {
        fill("#20b2aa");
        noStroke();
        ellipse(keypoint.position.x, keypoint.position.y, 10, 10);
      }
    }
  }
}

// A function to draw the skeletons
function drawSkeleton() {
  // Loop through all the skeletons detected
  for (let i = 0; i < poses.length; i++) {
    let skeleton = poses[i].skeleton;
    // For every skeleton, loop through all body connections
    for (let j = 0; j < skeleton.length; j++) {
      let partA = skeleton[j][0];
      let partB = skeleton[j][1];
      stroke("#20b2aa");
      line(
        partA.position.x,
        partA.position.y,
        partB.position.x,
        partB.position.y
      );
    }
  }
}

function startAudio() {
  sample1.loop();
  sample2.loop();
  sample3.loop();
  sample4.loop();
  sample5.loop();
  // sample1.rate(random() + 0.5);
  // sample2.rate(random() + 0.5);
  // sample3.rate(random() + 0.5);
  // sample4.rate(random() + 0.5);
}

function stopAudio() {
  sample1.stop();
  sample2.stop();
  sample3.stop();
  sample4.stop();
  sample5.stop();
}
