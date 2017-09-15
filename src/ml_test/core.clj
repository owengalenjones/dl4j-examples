(ns ml-test.core
  (:require [nd4clj.matrix]
            [clojure.core.matrix :as m])
  (:import (java.io DataInputStream)
           (java.util.zip GZIPInputStream)
           ; builders
           (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder)
           (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder)
           (org.deeplearning4j.nn.conf.layers.OutputLayer$Builder)
           (org.deeplearning4j.nn.multilayer MultiLayerNetwork)

           ; algos
           (org.deeplearning4j.nn.api OptimizationAlgorithm)
           (org.deeplearning4j.nn.weights WeightInit)
           (org.deeplearning4j.nn.conf.distribution UniformDistribution)

           ; matrix
           (org.nd4j.linalg.activations Activation)
           (org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction))
  (:gen-class))

;(let [file (fs/temp-file "ml")]
;  (try
;    #_(let [stream (GZIPInputStream. (io/resource "train-images-idx3-ubyte.gz"))]
;      (println "HI")
;      #_(with-open [input stream]
;        (loop [c (.read stream)]
;          (when (not= c -1)
;            (println c)
;            (recur (.read stream))))
;        ))
;      (finally
;        (fs/delete file))))

;(defmacro ub-to-double
;  "Convert an unsigned byte to a double, inline."
;  [item]
;  `(- (* ~item (double (/ 1.0 255.0)))
;      0.5))
;
;(with-open [s (DataInputStream. (GZIPInputStream. (io/input-stream (io/resource "train-images-idx3-ubyte.gz"))))]
;  (let [pic (image/new-image 28 28)
;        pixels (image/get-pixels pic)]
;    (println (.readInt s))
;    (println (.readInt s))
;    (println (.readInt s))
;    (println (.readInt s))
;    (dotimes [i (* 28 28)])
;      (image/set-pixels pic pixels)
;      (image/show pic)))
;
;(aset (byte-array 3) 1 "a")
;(colours/rand-colour)

(m/set-current-implementation :nd4j)

(def input (m/matrix [[0 0]
                      [1 0]
                      [0 1]
                      [1 1]]))
(def labels (m/matrix [[1 0]
                       [0 1]
                       [0 1]
                       [1 0]]))
(def dataset (DataSet. (.a input) (.a labels)))

(def builder (-> (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)
                 (.iterations 10000)
                 (.learningRate 0.1)
                 (.seed 123)
                 (.useDropConnect false)
                 (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                 (.biasInit 0)
                 (.miniBatch false)))

(def list-builder (.list builder))

(def hidden-layer-builder (-> (org.deeplearning4j.nn.conf.layers.DenseLayer$Builder.)
                              (.nIn 2)
                              (.nOut 4)
                              (.activation Activation/SIGMOID)
                              (.weightInit WeightInit/DISTRIBUTION)
                              (.dist (UniformDistribution. 0 1))))

(.layer list-builder 0 (.build hidden-layer-builder))

(def output-layer-builder (-> (org.deeplearning4j.nn.conf.layers.OutputLayer$Builder. (org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction/NEGATIVELOGLIKELIHOOD))
                              (.nIn 4)
                              (.nOut 2)
                              (.activation Activation/SOFTMAX)
                              (.weightInit WeightInit/DISTRIBUTION)
                              (.dist (UniformDistribution. 0 1))))

(.layer list-builder 1 (.build output-layer-builder))

(def conf (.build (-> list-builder
                      (.pretrain false)
                      (.backprop true))))

(def net (doto (MultiLayerNetwork. conf) .init))

(.fit net dataset)
(.getFeatureMatrix dataset)
