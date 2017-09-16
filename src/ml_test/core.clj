(ns ml-test.core
  (:require [nd4clj.matrix]
            [clojure.core.matrix :as m])
  (:import (org.deeplearning4j.eval Evaluation)

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
           (org.nd4j.linalg.dataset DataSet)
           (org.nd4j.linalg.activations Activation)
           (org.nd4j.linalg.lossfunctions.LossFunctions$LossFunction))
  (:gen-class))

(m/set-current-implementation :nd4j)

; [first input, second input]
(def input (m/matrix [[0 0]
                      [1 0]
                      [0 1]
                      [1 1]]))

; [true, false]
(def labels (m/matrix [[1 0]
                       [0 1]
                       [0 1]
                       [1 0]]))

; .a unwraps nd4clj.matrix.clj-INDArray -> org.nd4j.linalg.cpu.nativecpu.INDArray
(def dataset (DataSet. (.a input) (.a labels)))

; setup network configuration
(def list-builder (.list (-> (org.deeplearning4j.nn.conf.NeuralNetConfiguration$Builder.)
                             (.iterations 10000)
                             (.learningRate 0.1)
                             (.seed 123)
                             (.useDropConnect false)
                             (.optimizationAlgo OptimizationAlgorithm/STOCHASTIC_GRADIENT_DESCENT)
                             (.biasInit 0)
                             (.miniBatch false))))

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

(def net (doto (MultiLayerNetwork. conf)
           .init
           (.fit dataset)))

(def output (.output net (.getFeatureMatrix dataset)))
(def evalu (doto (Evaluation. 2)
             (.eval (.getLabels dataset) output)))

(println evalu)
