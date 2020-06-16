/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.os.Trace;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Dictionary;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier.Device;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/** A classifier specialized to label images using TensorFlow Lite. */
public abstract class Classifier {
  private static final Logger LOGGER = new Logger();

  /** The model type used for classification. */
  public enum Model {
    CUSTOM_EFFICIENTNET,
    QUANTIZED_MOBILENET,
    QUANTIZED_EFFICIENTNET
  }

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    GPU
  }

  /** Number of results to show in the UI. */
  private static final int MAX_RESULTS = 3;

  public class ModelWrapper {
    public MappedByteBuffer tfliteModel;
    public int inputSizeX;
    public int inputSizeY;
    public int outputSizeX;
    public int outputSizeY;
    public Interpreter tflite;
    public TensorImage inputImageBuffer;
    public TensorBuffer inputBuffer;
    public TensorBuffer outputBuffer;
  };

  public ModelWrapper feat = new ModelWrapper();
  public ModelWrapper headA = new ModelWrapper();
  public ModelWrapper headB = new ModelWrapper();
  public ModelWrapper headC = new ModelWrapper();

  /** The loaded TensorFlow Lite model. */
  private MappedByteBuffer tfliteModelHead, tfliteModelFeat;

  /** Image size along the x axis. */
  //private final int imageSizeX;

  /** Image size along the y axis. */
  //private final int imageSizeY;

  //private final int featSizeX, featSizeY;

  /** Optional GPU delegate for accleration. */
  private GpuDelegate gpuDelegate = null;

  /** Optional NNAPI delegate for accleration. */
  private NnApiDelegate nnApiDelegate = null;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tfliteHead, tfliteFeat;

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** Labels corresponding to the output of the vision model. */
  private List<String> labels;
  public static Dictionary<String, String> L2Map;
  public String gMode = "ABC";

  /** Input image TensorBuffer. */
  //private TensorImage inputImageBuffer;

  /** Output probability TensorBuffer. */
  //private final TensorBuffer outputProbabilityBuffer;

  //private final TensorBuffer inputFeatureBuffer, outputFeatureBuffer;

  /** Processer to apply post processing of the output probability. */
  private final TensorProcessor probabilityProcessor;

  //private final TensorProcessor featureProcessor;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param model The model to use for classification.
   * @param device The device to use for classification.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, Model model, Device device, int numThreads,
                                  String mode)
      throws IOException {
    if (model == Model.CUSTOM_EFFICIENTNET) {
      return new ClassifierCustomEfficientNet(activity, device, numThreads, mode);
    } else {
      throw new UnsupportedOperationException();
    }
  }

  /** An immutable result returned by a Classifier describing what was recognized. */
  public static class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /** Optional location within the source image for the location of the recognized object. */
    private RectF location;

    public Recognition(
        final String id, final String title, final Float confidence, final RectF location) {
      this.id = id;
      this.title = title;
      this.confidence = confidence;
      this.location = location;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getConfidence() {
      return confidence;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (confidence != null) {
        resultString += String.format("(%.1f%%) ", confidence * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }
  }

  /** Initializes a {@code Classifier}. */
  protected Classifier(Activity activity, Device device, int numThreads,
                       String mode) throws IOException {
    String[] modelPaths = getModelPath().split(";");
    this.feat.tfliteModel = FileUtil.loadMappedFile(activity, modelPaths[0]);
    headA.tfliteModel = FileUtil.loadMappedFile(activity, modelPaths[1]);
    headB.tfliteModel = FileUtil.loadMappedFile(activity, modelPaths[2]);
    headC.tfliteModel = FileUtil.loadMappedFile(activity, modelPaths[3]);

    switch (device) {
      case NNAPI:
        nnApiDelegate = new NnApiDelegate();
        tfliteOptions.addDelegate(nnApiDelegate);
        break;
      case GPU:
        gpuDelegate = new GpuDelegate();
        tfliteOptions.addDelegate(gpuDelegate);
        break;
      case CPU:
        break;
    }
    tfliteOptions.setNumThreads(numThreads);
    feat.tflite = new Interpreter(feat.tfliteModel, tfliteOptions);
    headA.tflite = new Interpreter(headA.tfliteModel, tfliteOptions);
    headB.tflite = new Interpreter(headB.tfliteModel, tfliteOptions);
    headC.tflite = new Interpreter(headC.tfliteModel, tfliteOptions);

    gMode = mode.replaceAll(" ", "");
    // Loads labels out from the label file.
    labels = FileUtil.loadLabels(activity, getLabelPathDict().get(mode).toString());
    L2Map = new Hashtable<String, String>();
    InputStream input;
    AssetManager assetManager = activity.getAssets();
    input = assetManager.open("labels_to_L2.txt");
    int size = input.available();
    byte[] buffer = new byte[size];
    input.read(buffer);
    input.close();
    // byte buffer into a string
    String text = new String(buffer);
    String[] entries = text.split("\n");
    for(String entry: entries){
      L2Map.put(entry.split("\t")[0], entry.split("\t")[1]);
    }
    LOGGER.d("L2Map:" + L2Map.get("othersAB") + L2Map.get("ostrich"));


      //FEAT
    // Reads type and shape of input and output tensors, respectively.
    int imageTensorIndex = 0;
    int[] imageShape = feat.tflite.getInputTensor(imageTensorIndex).shape(); // {1, height, width, 3}
    LOGGER.d("Feat: input shape: (%d, %d, %d, %d)", imageShape[0], imageShape[1], imageShape[2],
             imageShape[3]);
    feat.inputSizeY = imageShape[1];
    feat.inputSizeX = imageShape[2];
    DataType imageDataType = feat.tflite.getInputTensor(imageTensorIndex).dataType();
    int featTensorIndex = 0;
    int[] featShape =
        feat.tflite.getOutputTensor(featTensorIndex).shape(); // {1, NUM_CLASSES}
    LOGGER.d("Feat: output shape: (%d, %d, %d, %d)", featShape[0], featShape[1],
            featShape[2], featShape[3]);
    DataType featDataType = feat.tflite.getOutputTensor(featTensorIndex).dataType();

    // Creates the input tensor.
    feat.inputImageBuffer = new TensorImage(imageDataType);

    // Creates the output tensor and its processor.
    feat.outputBuffer = TensorBuffer.createFixedSize(featShape, featDataType);

    //HEAD
    // Reads type and shape of input and output tensors, respectively.
    int inFeatTensorIndex = 0;
    int[] inFeatShape = headA.tflite.getInputTensor(inFeatTensorIndex).shape(); // {1, height, width, 3}
    LOGGER.d("Head: input shape: (%d, %d, %d, %d)", inFeatShape[0], inFeatShape[1], inFeatShape[2],
            inFeatShape[3]);
    headA.inputSizeX = headB.inputSizeX = headC.inputSizeX = inFeatShape[1];
    headA.inputSizeY = headB.inputSizeY = headC.inputSizeY = inFeatShape[2];
    DataType inFeatDataType = headA.tflite.getInputTensor(inFeatTensorIndex).dataType();

    int probabilityTensorIndex = 0;
    int[] probabilityShape =
            headA.tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
    LOGGER.d("HeadA: output shape: (%d, %d)", probabilityShape[0], probabilityShape[1]);
    DataType probabilityDataType = headA.tflite.getOutputTensor(probabilityTensorIndex).dataType();
    //headA.inputBuffer = TensorBuffer.createFixedSize(inFeatShape, inFeatDataType);
    headA.outputBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

    probabilityTensorIndex = 0;
    probabilityShape =
            headB.tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
    LOGGER.d("HeadB: output shape: (%d, %d)", probabilityShape[0], probabilityShape[1]);
    probabilityDataType = headB.tflite.getOutputTensor(probabilityTensorIndex).dataType();
    //headB.inputBuffer = TensorBuffer.createFixedSize(inFeatShape, inFeatDataType);
    headB.outputBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

    probabilityTensorIndex = 0;
    probabilityShape =
            headC.tflite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
    LOGGER.d("Head: output shape: (%d, %d)", probabilityShape[0], probabilityShape[1]);
    probabilityDataType = headC.tflite.getOutputTensor(probabilityTensorIndex).dataType();
    //headC.inputBuffer = TensorBuffer.createFixedSize(inFeatShape, inFeatDataType);
    headC.outputBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

    // Creates the post processor for the output probability.
    probabilityProcessor = new TensorProcessor.Builder().add(getPostprocessNormalizeOp()).build();


    LOGGER.d("Created a Tensorflow Lite Image Classifier.");
  }

  /** Runs inference and returns the classification results. */
  public List<Recognition> recognizeImage(final Bitmap bitmap, int sensorOrientation) {


    float a[] = {-1, -2, -4, 9, 40};
    float b[] = {5, 7, -0, 20, 40};
    getProcessedOutput(a, b);

    float aa[] = {-1, -2, -4, 9, 40};
    float bb[] = {5, 7, -0, 20, 0};
    getProcessedOutput(aa, bb);


    float aaa[] = {-1, -2, -4, 9, 0};
    float bbb[] = {5, 7, -0, 20, 40};
    getProcessedOutput(aaa, bbb);


    float aaaa[] = {-1, -2, -4, 9, 0};
    float bbbb[] = {5, 7, -0, 20, 0};
    getProcessedOutput(aaaa, bbbb);


    float aaaaa[] = {-1, -2, -4, 9, 0};
    float bbbbb[] = {5, 7, -0, 20, 0};
    getProcessedOutput(aaaaa, bbbbb);



    // Logs this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");

    Trace.beginSection("loadImage");
    LOGGER.v("recognizeImage(E): Loading image");
    long startTimeForLoadImage = SystemClock.uptimeMillis();
    feat.inputImageBuffer = loadImage(bitmap, sensorOrientation);
    long endTimeForLoadImage = SystemClock.uptimeMillis();
    Trace.endSection();
    LOGGER.d("Timecost to load the image: " + (endTimeForLoadImage - startTimeForLoadImage));

    // Runs the inference call.
    Trace.beginSection("runInference");
    Trace.beginSection("featureExtraction");
    LOGGER.d("recognizeImage start featureExtraction");

    long startTimeForReference = SystemClock.uptimeMillis();
    feat.tflite.run(feat.inputImageBuffer.getBuffer(), feat.outputBuffer.getBuffer().rewind());
    long endTimeForReference = SystemClock.uptimeMillis();
    Trace.endSection();
    LOGGER.d("Timecost to run featureExtraction: " + (endTimeForReference - startTimeForReference));

    LOGGER.d("recognizeImage start conv head");
    startTimeForReference = SystemClock.uptimeMillis();
    TensorBuffer newProbBuffer = null;
    LOGGER.d("gMode: " + gMode );
    if(gMode.equals("A")) {
      LOGGER.d("Running in mode A");
      Trace.beginSection("running conv head");
      headA.tflite.run(feat.outputBuffer.getBuffer(), headA.outputBuffer.getBuffer().rewind());
      newProbBuffer = headA.outputBuffer;
    }
    else if(gMode.equals("B")) {
      LOGGER.d("Running in mode B");
      Trace.beginSection("running conv head");
      headB.tflite.run(feat.outputBuffer.getBuffer(), headB.outputBuffer.getBuffer().rewind());
      newProbBuffer = headB.outputBuffer;
    }
    else if(gMode.equals("C")) {
      LOGGER.d("Running in mode C");
      Trace.beginSection("running conv head");
      headC.tflite.run(feat.outputBuffer.getBuffer(), headC.outputBuffer.getBuffer().rewind());
      newProbBuffer = headC.outputBuffer;
    }
    else if(gMode.equals("AB")) {
      LOGGER.d("Running in mode AB");
      headA.tflite.run(feat.outputBuffer.getBuffer(), headA.outputBuffer.getBuffer().rewind());
      headB.tflite.run(feat.outputBuffer.getBuffer(), headB.outputBuffer.getBuffer().rewind());
      float[] srcA = headA.outputBuffer.getFloatArray();
      float[] srcB = headB.outputBuffer.getFloatArray();
      float[] src = getProcessedOutput(srcA, srcB);
      /*
      float[] src = new float[srcA.length - 1 + srcB.length - 1];
      for(int i = 0; i < srcA.length - 1; i++)
        src[i] = srcA[i];
      for(int i = 0; i < srcB.length - 1; i++)
        src[i + srcA.length - 1] = srcB[i];
      */
      LOGGER.d("src1: %d", srcA.length);
      LOGGER.d("src2: %d", srcB.length);
      LOGGER.d("concat: %f %f %f %f", src[0], src[1], src[2], src[3]);
      LOGGER.d("concat size %d", src.length);
      int[] new_shape = {1, src.length};

      newProbBuffer = TensorBuffer.createFixedSize(new_shape, headA.outputBuffer.getDataType());
      LOGGER.d("buffer shape %d, limit: %d", newProbBuffer.getShape()[0], newProbBuffer.getBuffer().limit());
      //LOGGER.d("temp buffer shape %d, limit: %d", new_buffer.capacity(), new_buffer.limit());
      newProbBuffer = TensorBuffer.createFixedSize(new_shape, headA.outputBuffer.getDataType());
      newProbBuffer.loadArray(src);
    }
    else if(gMode.equals("AC")) {
      LOGGER.d("Running in mode AC");
      headA.tflite.run(feat.outputBuffer.getBuffer(), headA.outputBuffer.getBuffer().rewind());
      headB.tflite.run(feat.outputBuffer.getBuffer(), headC.outputBuffer.getBuffer().rewind());
      float[] srcA = headA.outputBuffer.getFloatArray();
      float[] srcC = headC.outputBuffer.getFloatArray();
      float[] src = getProcessedOutput(srcA, srcC);
      LOGGER.d("src1: %d", srcA.length);
      LOGGER.d("src2: %d", srcC.length);
      LOGGER.d("concat: %f %f %f %f", src[0], src[1], src[2], src[3]);
      LOGGER.d("concat size %d", src.length);
      int[] new_shape = {1, src.length};
      newProbBuffer = TensorBuffer.createFixedSize(new_shape, headA.outputBuffer.getDataType());
      LOGGER.d("buffer shape %d, limit: %d", newProbBuffer.getShape()[0], newProbBuffer.getBuffer().limit());
      //LOGGER.d("temp buffer shape %d, limit: %d", new_buffer.capacity(), new_buffer.limit());
      newProbBuffer = TensorBuffer.createFixedSize(new_shape, headA.outputBuffer.getDataType());
      newProbBuffer.loadArray(src);
    }
    else if(gMode.equals("BC")) {
      LOGGER.d("Running in mode BC");
      headB.tflite.run(feat.outputBuffer.getBuffer(), headB.outputBuffer.getBuffer().rewind());
      headC.tflite.run(feat.outputBuffer.getBuffer(), headC.outputBuffer.getBuffer().rewind());
      float[] srcB = headB.outputBuffer.getFloatArray();
      float[] srcC = headC.outputBuffer.getFloatArray();
      float[] src = getProcessedOutput(srcB, srcC);
      LOGGER.d("src1: %d", srcB.length);
      LOGGER.d("src2: %d", srcC.length);
      LOGGER.d("concat: %f %f %f %f", src[0], src[1], src[2], src[3]);
      LOGGER.d("orig: %f %f %f %f", headA.outputBuffer.getFloatValue(0),
              headA.outputBuffer.getFloatValue(1),
              headA.outputBuffer.getFloatValue(2),
              headA.outputBuffer.getFloatValue(3));
      LOGGER.d("concat size %d", src.length);
      int[] new_shape = {1, src.length};
      newProbBuffer = TensorBuffer.createFixedSize(new_shape, headA.outputBuffer.getDataType());
      LOGGER.d("buffer shape %d, limit: %d", newProbBuffer.getShape()[0], newProbBuffer.getBuffer().limit());
      //LOGGER.d("temp buffer shape %d, limit: %d", new_buffer.capacity(), new_buffer.limit());
      newProbBuffer = TensorBuffer.createFixedSize(new_shape, headA.outputBuffer.getDataType());
      newProbBuffer.loadArray(src);
    }
    else {
        LOGGER.d("Running in mode ABC");
        headA.tflite.run(feat.outputBuffer.getBuffer(), headA.outputBuffer.getBuffer().rewind());
        headB.tflite.run(feat.outputBuffer.getBuffer(), headB.outputBuffer.getBuffer().rewind());
        headC.tflite.run(feat.outputBuffer.getBuffer(), headC.outputBuffer.getBuffer().rewind());
        /*
        float[] srcA = Arrays.copyOfRange(headA.outputBuffer.getFloatArray(), 0, 301);
        float[] srcB = Arrays.copyOfRange(headB.outputBuffer.getFloatArray(), 301, 601);
        float[] srcC = Arrays.copyOfRange(headC.outputBuffer.getFloatArray(), 601, 1000);
        float[] src = new float[srcA.length + srcB.length + srcC.length];
        */

        float[] srcA = headA.outputBuffer.getFloatArray();
        float[] srcB = headB.outputBuffer.getFloatArray();
        float[] srcC = headC.outputBuffer.getFloatArray();
        float[] src = new float[srcA.length - 1 + srcB.length - 1 + srcC.length - 1];
        for(int i = 0; i < srcA.length - 1; i++)
          src[i] = srcA[i];
        for(int i = 0; i < srcB.length - 1; i++)
          src[i + srcA.length - 1] = srcB[i];
        for(int i = 0; i < srcC.length - 1; i++)
          src[i + srcA.length - 1 + srcB.length - 1] = srcC[i];

        LOGGER.d("src1: %d", srcA.length);
        LOGGER.d("src2: %d", srcB.length);
        LOGGER.d("src3: %d", srcC.length);
        LOGGER.d("concat: %f %f %f %f", src[0], src[1], src[2], src[3]);
        LOGGER.d("orig: %f %f %f %f", headA.outputBuffer.getFloatValue(0),
                headA.outputBuffer.getFloatValue(1),
                headA.outputBuffer.getFloatValue(2),
                headA.outputBuffer.getFloatValue(3));

        LOGGER.d("concat size %d", src.length);
        int[] new_shape = {1, src.length};
        newProbBuffer = TensorBuffer.createFixedSize(new_shape, headA.outputBuffer.getDataType());
        LOGGER.d("buffer shape %d, limit: %d", newProbBuffer.getShape()[0], newProbBuffer.getBuffer().limit());
        //LOGGER.d("temp buffer shape %d, limit: %d", new_buffer.capacity(), new_buffer.limit());
        newProbBuffer = TensorBuffer.createFixedSize(new_shape, headA.outputBuffer.getDataType());
        newProbBuffer.loadArray(src);
    }

    endTimeForReference = SystemClock.uptimeMillis();
    Trace.endSection();
    LOGGER.d("Timecost to run convHead: " + (endTimeForReference - startTimeForReference));

    // Gets the map of label and probability.
    Map<String, Float> labeledProbability =
        new TensorLabel(labels, probabilityProcessor.process(newProbBuffer))
            .getMapWithFloatValue();
    Trace.endSection();

    // Gets top-k results.
    LOGGER.v("recognizeImage(X)");
    return getTopKProbability(labeledProbability);
  }

  private float[] getProcessedOutput(float[] srcA, float[] srcB) {

    int lenA = srcA.length;
    int lenB = srcB.length;
    String a = "";
    /*
    for(int i = 0; i < lenA; i++) {
      a += String.valueOf(srcA[i]) + " ";
    }

    String b = "";
    for(int i = 0; i < lenB; i++) {
      b += String.valueOf(srcB[i]) + " ";
    }

    //LOGGER.d("srcA: " + a);
    //LOGGER.d("srcB: " + b);
    */

    float[] src = new float[lenA + lenB - 1];

    float maxA = -10, maxB = -10;
    int maxAInd = 0, maxBInd = 0;
    for(int i = 0; i < lenA; i++) {
      src[i] = srcA[i];
      if(srcA[i] > maxA) {
        maxA = srcA[i];
        maxAInd = i;
      }
    }
    for(int i = 0; i < lenB; i++) {
      src[i + lenA - 1] = srcB[i];
      if(srcB[i] > maxB) {
        maxB = srcB[i];
        maxBInd = i;
      }
    }


    if(maxAInd == lenA - 1 && maxBInd == lenB - 1) {
      src[lenA + lenB - 2] = 10;
    }
    else if(maxAInd == lenA - 1) {
      for(int i = 0; i < lenA - 1; i++) {
        src[i] = -10;
      }
    }
    else if(maxBInd == lenB - 1) {
      for(int i = 0; i < lenB; i++) {
        src[i + lenA - 1] = -10;
      }
    }
    else {
      src[lenA + lenB - 2] = -10;
    }

    /*
    String c = "";
    for(int i = 0; i < src.length; i++) {
      c += String.valueOf(src[i]) + " ";
    }
    //LOGGER.d("src: " + c);
    */

    return src;
  }

  /** Closes the interpreter and model to release resources. */
  public void close() {
    if (tfliteHead != null) {
      tfliteHead.close();
      tfliteHead = null;
    }
    if (tfliteFeat != null) {
      tfliteFeat.close();
      tfliteFeat = null;
    }
    if (gpuDelegate != null) {
      gpuDelegate.close();
      gpuDelegate = null;
    }
    if (nnApiDelegate != null) {
      nnApiDelegate.close();
      nnApiDelegate = null;
    }
    tfliteModelFeat = null;
    tfliteModelHead = null;
  }

  /** Get the image size along the x axis. */
  public int getImageSizeX() {
    return feat.inputSizeX;
  }

  /** Get the image size along the y axis. */
  public int getImageSizeY() {
    return feat.inputSizeY;
  }

  /** Loads input image, and applies preprocessing. */
  private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
    // Loads bitmap into a TensorImage.
    feat.inputImageBuffer.load(bitmap);

    // Creates processor for the TensorImage.
    int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
    int numRotation = sensorOrientation / 90;
    // TODO(b/143564309): Fuse ops inside ImageProcessor.
    ImageProcessor imageProcessor =
        new ImageProcessor.Builder()
            .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(new ResizeOp(feat.inputSizeX, feat.inputSizeY, ResizeMethod.NEAREST_NEIGHBOR))
            .add(new Rot90Op(numRotation))
            .add(getPreprocessNormalizeOp())
            .build();
    return imageProcessor.process(feat.inputImageBuffer);
  }

  /** Gets the top-k results. */
  private static List<Recognition> getTopKProbability(Map<String, Float> labelProb) {
    // Find the best classifications.
    PriorityQueue<Recognition> pq =
        new PriorityQueue<>(
            MAX_RESULTS,
            new Comparator<Recognition>() {
              @Override
              public int compare(Recognition lhs, Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
              }
            });

    for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
      pq.add(new Recognition("" + entry.getKey(), L2Map.get(entry.getKey()), entry.getValue(), null));
    }

    final ArrayList<Recognition> recognitions = new ArrayList<>();
    int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
    for (int i = 0; i < recognitionsSize; ++i) {
      recognitions.add(pq.poll());
    }
    return recognitions;
  }

  /** Gets the name of the model file stored in Assets. */
  protected abstract String getModelPath();

  /** Gets the name of the label file stored in Assets. */
  protected abstract String getLabelPath();

  protected Dictionary getLabelPathDict() {
    return null;
  };


  /** Gets the TensorOperator to nomalize the input image in preprocessing. */
  protected abstract TensorOperator getPreprocessNormalizeOp();

  /**
   * Gets the TensorOperator to dequantize the output probability in post processing.
   *
   * <p>For quantized model, we need de-quantize the prediction with NormalizeOp (as they are all
   * essentially linear transformation). For float model, de-quantize is not required. But to
   * uniform the API, de-quantize is added to float model too. Mean and std are set to 0.0f and
   * 1.0f, respectively.
   */
  protected abstract TensorOperator getPostprocessNormalizeOp();
}
